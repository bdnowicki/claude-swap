"""Display names (`cswap name`): validation, store, CLI, list, JSON, transfer.

A display name is the free-form sibling of the alias: shown in front of it
("Name · alias (email)"), never used to resolve an account. Everything here
runs against a throwaway store under `temp_home`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli, oauth
from claude_swap.credentials import ActiveCredentials
from claude_swap.exceptions import AccountNotFoundError, ValidationError
from claude_swap.models import (
    DISPLAY_NAME_MAX_LEN,
    account_label,
    normalize_display_name,
)
from claude_swap.switcher import ClaudeAccountSwitcher, _styled_account_label
from claude_swap.transfer import TransferError, export_accounts, import_accounts
from tests.test_transfer import _linux_switcher, _seed_account

EN_DASH = chr(0x2013)
ZWJ = chr(0x200D)


# --- the rule itself -----------------------------------------------------------


class TestNormalizeDisplayName:
    def test_trims_and_keeps_case_spaces_and_unicode(self):
        assert normalize_display_name(f"  Luxoft {EN_DASH} main  ") == f"Luxoft {EN_DASH} main"

    def test_composite_emoji_survive(self):
        # The joiners are Unicode "format" characters, refused in general
        # (bidi overrides are too), but emoji are built from them.
        coder = chr(0x1F468) + ZWJ + chr(0x1F4BB)
        assert normalize_display_name(f"dev {coder}") == f"dev {coder}"

    def test_the_limit_is_inclusive(self):
        assert normalize_display_name("x" * DISPLAY_NAME_MAX_LEN) == "x" * DISPLAY_NAME_MAX_LEN
        with pytest.raises(ValueError, match="limit is 40"):
            normalize_display_name("x" * (DISPLAY_NAME_MAX_LEN + 1))

    @pytest.mark.parametrize(
        "bad",
        ["", "   ", "a\nb", "a\tb", "a" + chr(0x202E) + "b", "a" + chr(0x2028) + "b"],
        ids=["empty", "blank", "newline", "tab", "bidi-override", "line-separator"],
    )
    def test_refuses_what_would_break_a_one_line_header(self, bad):
        with pytest.raises(ValueError):
            normalize_display_name(bad)


class TestAccountLabel:
    @pytest.mark.parametrize(
        ("alias", "name", "expected"),
        [
            ("", "", "a@x.com"),
            ("dev", "", "dev (a@x.com)"),  # exactly the format before names
            ("", "Main", "Main (a@x.com)"),
            ("dev", "Main", "Main · dev (a@x.com)"),
        ],
    )
    def test_every_combination(self, alias, name, expected):
        assert account_label("a@x.com", alias, name) == expected

    def test_styled_list_label_is_the_plain_one_without_colour(self, monkeypatch):
        from claude_swap import printer

        monkeypatch.setattr(printer, "colors_enabled", lambda: False)
        for alias, name in (("", ""), ("dev", ""), ("", "Main"), ("dev", "Main")):
            assert _styled_account_label("a@x.com", alias, name) == account_label(
                "a@x.com", alias, name
            )


# --- the store -------------------------------------------------------------------


class TestSwitcherDisplayName:
    def _switcher(self, sample_sequence_data):
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, sample_sequence_data)
        return switcher

    def test_set_by_number_alias_and_email(self, temp_home, sample_sequence_data):
        sample_sequence_data["accounts"]["2"]["alias"] = "dev"
        switcher = self._switcher(sample_sequence_data)
        assert switcher.set_display_name("1", " First ") == ("1", "First")
        assert switcher.set_display_name("dev", "Second") == ("2", "Second")
        assert switcher.set_display_name("account1@example.com", "Again") == ("1", "Again")
        accounts = switcher._get_sequence_data()["accounts"]
        assert accounts["1"]["displayName"] == "Again"
        assert accounts["2"]["displayName"] == "Second"

    def test_names_need_not_be_unique(self, temp_home, sample_sequence_data):
        switcher = self._switcher(sample_sequence_data)
        switcher.set_display_name("1", "Luxoft")
        switcher.set_display_name("2", "Luxoft")
        assert [n for _, n, _ in switcher.list_display_names()] == ["Luxoft", "Luxoft"]

    def test_a_name_never_resolves_an_account(self, temp_home, sample_sequence_data):
        """Display only: switch/remove/run must not start matching it."""
        switcher = self._switcher(sample_sequence_data)
        switcher.set_display_name("2", "work")
        assert switcher._resolve_account_identifier("work") is None

    def test_invalid_name_raises_validation_error(self, temp_home, sample_sequence_data):
        switcher = self._switcher(sample_sequence_data)
        with pytest.raises(ValidationError):
            switcher.set_display_name("1", "a\nb")
        assert "displayName" not in switcher._get_sequence_data()["accounts"]["1"]

    def test_unknown_account_raises(self, temp_home, sample_sequence_data):
        switcher = self._switcher(sample_sequence_data)
        with pytest.raises(AccountNotFoundError):
            switcher.set_display_name("9", "x")
        with pytest.raises(AccountNotFoundError):
            switcher.unset_display_name("9")

    def test_unset_is_idempotent(self, temp_home, sample_sequence_data):
        switcher = self._switcher(sample_sequence_data)
        switcher.set_display_name("1", "First")
        assert switcher.unset_display_name("1") == "1"
        assert switcher.unset_display_name("1") == "1"
        assert "displayName" not in switcher._get_sequence_data()["accounts"]["1"]

    def test_list_is_in_slot_order_and_skips_junk(self, temp_home, sample_sequence_data):
        sample_sequence_data["accounts"]["1"]["displayName"] = ["hand", "edited"]
        switcher = self._switcher(sample_sequence_data)
        switcher.set_display_name("2", "Second")
        assert switcher.list_display_names() == [("2", "Second", "account2@example.com")]
        # ...and the snapshot reader degrades the same way: no label, no crash.
        data = switcher._get_sequence_data()
        assert switcher._display_name_from_data(data, "1") == ""

    def test_move_carries_the_name(self, temp_home, sample_sequence_data):
        switcher = self._switcher(sample_sequence_data)
        switcher.set_display_name("2", "Second")
        with patch.object(switcher, "_live_session_pids", return_value=[]):
            switcher.move_account("2", "5")
        accounts = switcher._get_sequence_data()["accounts"]
        assert accounts["5"]["displayName"] == "Second"
        assert "2" not in accounts


class TestReaddKeepsTheName:
    """`cswap add` rebuilds the account record; the name must ride along
    exactly like the alias does (mirrors TestAddAccountAlias)."""

    def _switcher(self, temp_home, email):
        config = {
            "oauthAccount": {
                "emailAddress": email,
                "accountUuid": "uuid-" + email,
                "organizationUuid": "",
                "organizationName": "",
            }
        }
        (temp_home / ".claude.json").write_text(json.dumps(config))
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._init_sequence_file()
        return switcher

    def _add(self, switcher, **kwargs):
        creds = json.dumps({"claudeAiOauth": {"accessToken": "tok"}})
        with patch.object(
            switcher, "_read_active_credentials",
            return_value=ActiveCredentials(creds, False),
        ), patch.object(switcher, "_write_account_credentials"), \
             patch.object(switcher, "_delete_account_credentials"):
            switcher.add_account(**kwargs)

    def test_refresh_in_place_keeps_it(self, temp_home):
        switcher = self._switcher(temp_home, "a@x.com")
        self._add(switcher)
        switcher.set_display_name("1", "Main")
        self._add(switcher)
        assert switcher._get_sequence_data()["accounts"]["1"]["displayName"] == "Main"

    def test_readding_into_the_same_explicit_slot_keeps_it(self, temp_home):
        switcher = self._switcher(temp_home, "a@x.com")
        self._add(switcher)
        switcher.set_display_name("1", "Main")
        self._add(switcher, slot=1)
        assert switcher._get_sequence_data()["accounts"]["1"]["displayName"] == "Main"

    def test_slot_migration_carries_it(self, temp_home):
        switcher = self._switcher(temp_home, "a@x.com")
        self._add(switcher)
        switcher.set_display_name("1", "Main")
        self._add(switcher, slot=5)
        accounts = switcher._get_sequence_data()["accounts"]
        assert "1" not in accounts
        assert accounts["5"]["displayName"] == "Main"


# --- what people see ----------------------------------------------------------------


class TestListShowsTheName:
    def _list(self, sample_sequence_data, *, json_output=False):
        active_creds = json.dumps({"claudeAiOauth": {"accessToken": "sk-active"}})
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, sample_sequence_data)
        with patch.object(
            switcher, "_read_active_credentials",
            return_value=ActiveCredentials(active_creds, False),
        ), patch.object(switcher, "_read_account_credentials", return_value=""), \
             patch(
                 "claude_swap.oauth.try_fetch_usage_for_account",
                 return_value=oauth.UsageOutcome(None),
             ):
            return switcher.list_accounts(json_output=json_output)

    def test_human_header_leads_with_the_name(
        self, temp_home, mock_claude_config, sample_sequence_data, capsys
    ):
        accounts = sample_sequence_data["accounts"]
        accounts["1"].update(email="test@example.com", alias="dev", displayName="Main")
        accounts["2"]["displayName"] = "Second"
        self._list(sample_sequence_data)
        out = capsys.readouterr().out
        assert "  1: Main · dev (test@example.com) [personal] (active)" in out
        assert "  2: Second (account2@example.com) [personal]" in out

    def test_json_carries_display_name_only_when_set(
        self, temp_home, mock_claude_config, sample_sequence_data
    ):
        sample_sequence_data["accounts"]["1"].update(
            email="test@example.com", displayName="Main"
        )
        payload = self._list(sample_sequence_data, json_output=True)
        by_num = {a["number"]: a for a in payload["accounts"]}
        assert by_num[1]["displayName"] == "Main"
        assert "displayName" not in by_num[2]


# --- the CLI --------------------------------------------------------------------------


class TestNameCommand:
    def _seed(self, temp_home):
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._init_sequence_file()
        data = switcher._get_sequence_data()
        data["accounts"]["2"] = {
            "email": "work@co.com", "uuid": "u2", "organizationUuid": "",
            "organizationName": "", "added": "2024-01-01T00:00:00Z", "alias": "dev",
        }
        data["sequence"] = [2]
        switcher._write_json(switcher.sequence_file, data)

    def _run(self, argv):
        with patch("os.geteuid", return_value=1000, create=True):
            cli._name_command(argv)

    def _name(self):
        return ClaudeAccountSwitcher()._get_sequence_data()["accounts"]["2"].get("displayName")

    def test_several_words_join_without_quotes(self, temp_home, capsys):
        self._seed(temp_home)
        self._run(["dev", "Work", "laptop"])
        assert self._name() == "Work laptop"
        assert "Work laptop" in capsys.readouterr().out

    def test_one_quoted_argument(self, temp_home):
        self._seed(temp_home)
        self._run(["2", f"Work {EN_DASH} laptop"])
        assert self._name() == f"Work {EN_DASH} laptop"

    def test_unset_and_list(self, temp_home, capsys):
        self._seed(temp_home)
        self._run(["2", "Work"])
        self._run([])
        assert "2: Work" in capsys.readouterr().out
        self._run(["2", "--unset"])
        assert self._name() is None
        self._run([])
        assert "No names set" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "argv",
        [["2"], ["--unset"], ["2", "Work", "--unset"]],
        ids=["no-text", "unset-without-account", "unset-with-text"],
    )
    def test_usage_errors(self, temp_home, argv):
        self._seed(temp_home)
        with pytest.raises(SystemExit) as exc:
            self._run(argv)
        assert exc.value.code == 2

    def test_too_long_is_an_error_not_a_truncation(self, temp_home, capsys):
        self._seed(temp_home)
        with pytest.raises(SystemExit) as exc:
            self._run(["2", "x" * (DISPLAY_NAME_MAX_LEN + 1)])
        assert exc.value.code == 1
        assert "limit is 40" in capsys.readouterr().err
        assert self._name() is None

    def test_dispatched_from_main(self, temp_home):
        with patch("claude_swap.cli._name_command") as name_fn, \
             patch.object(sys, "argv", ["claude-swap", "name", "2", "Work"]):
            cli.main()
        name_fn.assert_called_once_with(["2", "Work"])

    def test_main_help_lists_it(self, temp_home, capsys):
        with patch.object(sys, "argv", ["claude-swap", "help"]):
            with pytest.raises(SystemExit):
                cli.main()
        assert "name <num|email|alias> <text>" in capsys.readouterr().out


# --- export / import ----------------------------------------------------------------------


class TestNameTransfer:
    def _import_into_fresh_home(self, temp_home: Path, out_file: Path, **seed_dst):
        dst_home = temp_home.parent / "dst"
        dst_home.mkdir()
        with patch("pathlib.Path.home", return_value=dst_home), \
             patch.dict(os.environ, {"HOME": str(dst_home)}):
            dst = _linux_switcher(dst_home)
            import_accounts(dst, str(out_file))
            return dst._get_sequence_data()

    def test_round_trips(self, temp_home):
        src = _linux_switcher(temp_home)
        _seed_account(src, 1, "alice@example.com", alias="dev")
        _seed_account(src, 2, "bob@example.com")
        src.set_display_name("1", f"Alice {EN_DASH} main")

        out_file = temp_home / "backup.cswap"
        export_accounts(src, str(out_file))
        by_email = {a["email"]: a for a in json.loads(out_file.read_text())["accounts"]}
        assert by_email["alice@example.com"]["displayName"] == f"Alice {EN_DASH} main"
        assert "displayName" not in by_email["bob@example.com"]

        seq = self._import_into_fresh_home(temp_home, out_file)
        assert seq["accounts"]["1"]["displayName"] == f"Alice {EN_DASH} main"
        assert seq["accounts"]["1"]["alias"] == "dev"
        assert "displayName" not in seq["accounts"]["2"]

    def test_an_invalid_name_fails_the_import_before_any_write(self, temp_home):
        src = _linux_switcher(temp_home)
        _seed_account(src, 1, "alice@example.com")
        out_file = temp_home / "backup.cswap"
        export_accounts(src, str(out_file))
        envelope = json.loads(out_file.read_text())
        envelope["accounts"][0]["displayName"] = "two\nlines"
        out_file.write_text(json.dumps(envelope))

        with pytest.raises(TransferError, match="invalid displayName"):
            self._import_into_fresh_home(temp_home, out_file)

    def test_a_non_string_name_fails_the_import(self, temp_home):
        src = _linux_switcher(temp_home)
        _seed_account(src, 1, "alice@example.com")
        out_file = temp_home / "backup.cswap"
        export_accounts(src, str(out_file))
        envelope = json.loads(out_file.read_text())
        envelope["accounts"][0]["displayName"] = 7
        out_file.write_text(json.dumps(envelope))

        with pytest.raises(TransferError, match="displayName"):
            self._import_into_fresh_home(temp_home, out_file)
