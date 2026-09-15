"""Tests for the oauth module."""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import oauth


class TestExtractAccessToken:
    """Test extract_access_token."""

    def test_valid_credentials(self):
        creds = json.dumps({"claudeAiOauth": {"accessToken": "sk-test-token"}})
        assert oauth.extract_access_token(creds) == "sk-test-token"

    def test_missing_key(self):
        creds = json.dumps({"claudeAiOauth": {}})
        assert oauth.extract_access_token(creds) is None

    def test_invalid_json(self):
        assert oauth.extract_access_token("not-json") is None

    def test_empty_string(self):
        assert oauth.extract_access_token("") is None


class TestAccountHeadroom:
    """Test account_headroom."""

    def test_binding_window_is_the_higher_utilization(self):
        usage = {"five_hour": {"pct": 80.0}, "seven_day": {"pct": 20.0}}
        assert oauth.account_headroom(usage) == 20.0  # 100 - max(80, 20)

    def test_seven_day_can_be_the_binding_window(self):
        usage = {"five_hour": {"pct": 10.0}, "seven_day": {"pct": 95.0}}
        assert oauth.account_headroom(usage) == 5.0

    def test_single_window(self):
        assert oauth.account_headroom({"five_hour": {"pct": 40.0}}) == 60.0

    def test_at_limit_is_zero_headroom(self):
        assert oauth.account_headroom({"five_hour": {"pct": 100.0}}) == 0.0

    def test_spend_is_ignored(self):
        # Pay-as-you-go credits must not drive rate-limit headroom.
        usage = {"spend": {"pct": 99.0}, "five_hour": {"pct": 10.0}}
        assert oauth.account_headroom(usage) == 90.0

    def test_spend_binds_when_the_account_has_no_rate_windows(self):
        # Was `test_no_window_data_is_unknown`'s first assertion, which
        # expected None. A dollar-budget account reports no 5h/7d/scoped at
        # all, so reading it as "unknown" is what made `cswap auto` fail over
        # off a perfectly healthy Enterprise account. Money is the only gate it
        # has, so money binds.
        assert oauth.account_headroom({"spend": {"pct": 50.0}}) == 50.0

    def test_no_window_data_is_unknown(self):
        # Genuinely empty usage stays unknown — never auto-skipped.
        assert oauth.account_headroom({}) is None

    def test_none_and_non_dict_are_unknown(self):
        assert oauth.account_headroom(None) is None
        assert oauth.account_headroom("no credentials") is None

    def test_malformed_pct_is_ignored(self):
        assert oauth.account_headroom({"five_hour": {"pct": None}}) is None

    def test_scoped_ignored_without_models_arg(self):
        # Default behavior is unchanged: per-model windows never bind.
        usage = {"five_hour": {"pct": 10.0}, "scoped": [{"name": "Fable", "pct": 100.0}]}
        assert oauth.account_headroom(usage) == 90.0

    def test_named_model_folds_into_binding_window(self):
        usage = {"five_hour": {"pct": 10.0}, "scoped": [{"name": "Fable", "pct": 95.0}]}
        assert oauth.account_headroom(usage, ["Fable"]) == 5.0

    def test_maxed_model_is_at_limit_despite_session_headroom(self):
        # The exact motivating case: 5h/7d fine, but the model is exhausted.
        usage = {
            "five_hour": {"pct": 1.0},
            "seven_day": {"pct": 40.0},
            "scoped": [{"name": "Fable", "pct": 100.0}],
        }
        assert oauth.account_headroom(usage, ["Fable"]) == 0.0

    def test_model_match_is_case_insensitive(self):
        usage = {"scoped": [{"name": "Fable", "pct": 70.0}]}
        assert oauth.account_headroom(usage, ["fable"]) == 30.0

    def test_unlisted_model_does_not_bind(self):
        usage = {"five_hour": {"pct": 10.0}, "scoped": [{"name": "Opus", "pct": 100.0}]}
        assert oauth.account_headroom(usage, ["Fable"]) == 90.0

    def test_multiple_models_take_the_worst(self):
        usage = {
            "five_hour": {"pct": 10.0},
            "scoped": [
                {"name": "Fable", "pct": 30.0},
                {"name": "Opus", "pct": 95.0},
                {"name": "Haiku", "pct": 50.0},
            ],
        }
        # Opus binds (95%); Sonnet is absent and simply contributes nothing.
        assert oauth.account_headroom(usage, ["Fable", "Opus", "Sonnet"]) == 5.0

    def test_works_for_any_model_name(self):
        for name in ("Opus", "Sonnet", "Haiku"):
            usage = {"scoped": [{"name": name, "pct": 100.0}]}
            assert oauth.account_headroom(usage, [name]) == 0.0

    def test_only_scoped_and_named_yields_headroom(self):
        # No 5h/7d at all (the live shape when the API returns only limits).
        assert oauth.account_headroom({"scoped": [{"name": "Fable", "pct": 100.0}]}, ["Fable"]) == 0.0

    def test_scoped_without_5h7d_and_unlisted_model_is_unknown(self):
        usage = {"scoped": [{"name": "Opus", "pct": 100.0}]}
        assert oauth.account_headroom(usage, ["Fable"]) is None

    def test_all_sentinel_matches_every_scoped_window(self):
        usage = {
            "five_hour": {"pct": 10.0},
            "scoped": [
                {"name": "Fable", "pct": 30.0},
                {"name": "Sonnet", "pct": 97.0},
            ],
        }
        assert oauth.account_headroom(usage, ["all"]) == 3.0
        assert oauth.account_headroom(usage, ["ALL"]) == 3.0


class TestRelevantWindows:
    """Test relevant_windows — the canonical window source."""

    def test_carries_labels_pcts_and_resets(self):
        usage = {
            "five_hour": {"pct": 80.0, "resets_at": "2026-07-10T12:00:00Z"},
            "seven_day": {"pct": 20.0},
            "scoped": [
                {"name": "Fable", "pct": 95.0, "resets_at": "2026-07-12T09:00:00Z"},
            ],
        }
        assert oauth.relevant_windows(usage, ["Fable"]) == [
            ("5h", 80.0, "2026-07-10T12:00:00Z"),
            ("7d", 20.0, None),
            ("Fable", 95.0, "2026-07-12T09:00:00Z"),
        ]

    def test_scoped_excluded_without_models(self):
        usage = {"five_hour": {"pct": 10.0}, "scoped": [{"name": "Fable", "pct": 99.0}]}
        assert oauth.relevant_windows(usage) == [("5h", 10.0, None)]

    def test_non_dict_usage_is_empty(self):
        assert oauth.relevant_windows(None) == []
        assert oauth.relevant_windows("no credentials") == []


class TestFormatReset:
    """Test format_reset."""

    def test_same_day_shows_time_only(self):
        from datetime import timedelta
        fixed_now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = fixed_now + timedelta(hours=2, minutes=15)
        with patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.now.return_value = fixed_now
            countdown, clock = oauth.format_reset(future.isoformat())
        assert countdown == "2h 15m"
        assert clock.count(":") == 1

    def test_different_day_shows_date(self):
        from datetime import timedelta
        fixed_now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = fixed_now + timedelta(days=2)
        with patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.now.return_value = fixed_now
            countdown, clock = oauth.format_reset(future.isoformat())
        import calendar
        months = list(calendar.month_abbr)[1:]
        assert any(m in clock for m in months)

    def test_minutes_only_when_under_one_hour(self):
        from datetime import timedelta
        fixed_now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = fixed_now + timedelta(minutes=45)
        with patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.now.return_value = fixed_now
            countdown, clock = oauth.format_reset(future.isoformat())
        assert countdown == "45m"
        assert "h" not in countdown


class TestFetchUsage:
    """Test fetch_usage."""

    def test_success(self):
        from datetime import timedelta
        fixed_now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = fixed_now + timedelta(hours=1)
        response_data = {
            "five_hour": {"utilization": 22.0, "resets_at": future.isoformat()},
            "seven_day": {"utilization": 61.0, "resets_at": future.isoformat()},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=mock_response), \
             patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.now.return_value = fixed_now
            result = oauth.fetch_usage("sk-test-token")

        assert result["five_hour"]["pct"] == 22.0
        assert result["seven_day"]["pct"] == 61.0
        assert result["five_hour"]["countdown"] == "1h 0m"

    def test_network_error(self):
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=Exception("timeout")):
            result = oauth.fetch_usage("sk-test-token")
        assert result is None

    def test_http_error_logs_in_debug_mode(self, capsys):
        import logging
        logger = logging.getLogger("claude-swap")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        try:
            http_error = urllib.error.HTTPError(
                url="https://api.anthropic.com/api/oauth/usage",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )

            with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=http_error):
                result = oauth.fetch_usage("sk-test-token")

            assert result is None
            debug_output = capsys.readouterr().err
            assert "Usage fetch failed" in debug_output
            assert "<HTTPError 429: 'Too Many Requests'>" in debug_output
        finally:
            logger.removeHandler(handler)
            logger.setLevel(logging.WARNING)

    def test_bad_response(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=mock_response):
            result = oauth.fetch_usage("sk-test-token")
        assert result is None

    def test_null_resets_at(self):
        """When resets_at is null, still return pct without clock/countdown."""
        from datetime import timedelta
        fixed_now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = fixed_now + timedelta(hours=22)
        response_data = {
            "five_hour": {"utilization": 0.0, "resets_at": None},
            "seven_day": {"utilization": 100.0, "resets_at": future.isoformat()},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=mock_response), \
             patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.now.return_value = fixed_now
            result = oauth.fetch_usage("sk-test-token")

        assert result is not None
        assert result["five_hour"]["pct"] == 0.0
        assert "clock" not in result["five_hour"]
        assert "countdown" not in result["five_hour"]
        assert result["seven_day"]["pct"] == 100.0
        assert "clock" in result["seven_day"]
        assert "countdown" in result["seven_day"]

    @staticmethod
    def _fetch_with_response(response_data):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=mock_response):
            return oauth.fetch_usage("sk-test-token")

    def test_extra_usage_complete(self):
        """All extra_usage fields populated — spend, five_hour, and seven_day all present."""
        result = self._fetch_with_response({
            "five_hour": {"utilization": 22.0, "resets_at": None},
            "seven_day": {"utilization": 61.0, "resets_at": None},
            "extra_usage": {
                "is_enabled": True,
                "used_credits": 72900,
                "monthly_limit": 500000,
                "utilization": 14.58,
                "currency": "USD",
            },
        })
        assert result is not None
        assert result["five_hour"]["pct"] == 22.0
        assert result["seven_day"]["pct"] == 61.0
        assert result["spend"]["used"] == 729.0
        assert result["spend"]["limit"] == 5000.0
        assert result["spend"]["pct"] == 14.58
        assert result["spend"]["currency"] == "USD"

    def test_extra_usage_unlimited_keeps_other_rows(self):
        """Unlimited (monthly_limit=None) drops the spend entry without losing five_hour/seven_day."""
        result = self._fetch_with_response({
            "five_hour": {"utilization": 22.0, "resets_at": None},
            "seven_day": {"utilization": 61.0, "resets_at": None},
            "extra_usage": {
                "is_enabled": True,
                "used_credits": 72900,
                "monthly_limit": None,
                "utilization": None,
                "currency": "USD",
            },
        })
        assert result is not None
        assert result["five_hour"]["pct"] == 22.0
        assert result["seven_day"]["pct"] == 61.0
        assert "spend" not in result

    def test_extra_usage_partial_keeps_other_rows(self):
        """A null in used_credits leaves the rest of the response untouched."""
        result = self._fetch_with_response({
            "five_hour": {"utilization": 22.0, "resets_at": None},
            "seven_day": {"utilization": 61.0, "resets_at": None},
            "extra_usage": {
                "is_enabled": True,
                "used_credits": None,
                "monthly_limit": 500000,
                "utilization": 14.58,
            },
        })
        assert result is not None
        assert result["five_hour"]["pct"] == 22.0
        assert result["seven_day"]["pct"] == 61.0
        assert "spend" not in result

    def test_extra_usage_disabled_keeps_other_rows(self):
        """is_enabled=False suppresses spend even with valid numeric fields."""
        result = self._fetch_with_response({
            "five_hour": {"utilization": 22.0, "resets_at": None},
            "seven_day": {"utilization": 61.0, "resets_at": None},
            "extra_usage": {
                "is_enabled": False,
                "used_credits": 72900,
                "monthly_limit": 500000,
                "utilization": 14.58,
            },
        })
        assert result is not None
        assert result["five_hour"]["pct"] == 22.0
        assert result["seven_day"]["pct"] == 61.0
        assert "spend" not in result

    def test_scoped_per_model_limits(self):
        """weekly_scoped entries in limits[] surface as result['scoped'] by model name."""
        from datetime import timedelta
        fixed_now = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = fixed_now + timedelta(hours=3)
        response_data = {
            "five_hour": {"utilization": 7.0, "resets_at": None},
            "seven_day": {"utilization": 72.0, "resets_at": None},
            "seven_day_opus": None,
            "limits": [
                {"kind": "session", "group": "session", "percent": 7,
                 "resets_at": None, "scope": None, "is_active": False},
                {"kind": "weekly_all", "group": "weekly", "percent": 72,
                 "resets_at": None, "scope": None, "is_active": False},
                {"kind": "weekly_scoped", "group": "weekly", "percent": 100,
                 "severity": "critical", "resets_at": future.isoformat(),
                 "scope": {"model": {"id": None, "display_name": "Fable"}, "surface": None},
                 "is_active": True},
            ],
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=mock_response), \
             patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.now.return_value = fixed_now
            result = oauth.fetch_usage("sk-test-token")

        assert result is not None
        # Only the model-scoped entry is surfaced; session/weekly_all (scope=None) are not.
        assert len(result["scoped"]) == 1
        fable = result["scoped"][0]
        assert fable["name"] == "Fable"
        assert fable["pct"] == 100.0
        assert fable["resets_at"] == future.isoformat()
        assert fable["countdown"] == "3h 0m"
        assert "clock" in fable

    def test_no_limits_no_scoped_key(self):
        """A response without a limits array yields no 'scoped' key (backward compat)."""
        result = self._fetch_with_response({
            "five_hour": {"utilization": 22.0, "resets_at": None},
            "seven_day": {"utilization": 61.0, "resets_at": None},
        })
        assert result is not None
        assert "scoped" not in result


class TestRefreshOAuthCredentials:
    """Test direct OAuth refresh requests."""

    @staticmethod
    def _make_credentials(scopes=None):
        if scopes is None:
            scopes = ["user:profile", "user:inference", "user:sessions:claude_code"]
        return json.dumps({
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
                "expiresAt": 0,
                "scopes": scopes,
            }
        })

    def test_refresh_sends_correct_body(self):
        seen_body = {}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        def mock_urlopen(req, timeout=0):
            seen_body.update(json.loads(req.data.decode()))
            return mock_response

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            refreshed = oauth.refresh_oauth_credentials(self._make_credentials())

        assert refreshed is not None
        assert seen_body["grant_type"] == "refresh_token"
        assert seen_body["refresh_token"] == "old-refresh"
        assert seen_body["client_id"] == oauth.OAUTH_CLIENT_ID
        assert "scope" not in seen_body


class TestTryRefreshOAuthCredentials:
    """Typed refresh outcomes: permanent vs transient failure classification."""

    _make_credentials = staticmethod(TestRefreshOAuthCredentials._make_credentials)

    @staticmethod
    def _http_error(code, body: bytes, msg="err"):
        import io

        return urllib.error.HTTPError(
            oauth.OAUTH_TOKEN_URL, code, msg, hdrs=None, fp=io.BytesIO(body)
        )

    def test_success_rotates_and_has_no_error(self):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(
            "claude_swap.oauth.urllib.request.urlopen", return_value=mock_response
        ):
            outcome = oauth.try_refresh_oauth_credentials(self._make_credentials())

        assert outcome.error is None
        rotated = json.loads(outcome.credentials)["claudeAiOauth"]
        assert rotated["accessToken"] == "new-access"
        assert rotated["refreshToken"] == "new-refresh"

    def test_invalid_grant_body_on_400_is_permanent(self):
        err = self._http_error(400, b'{"error": "invalid_grant"}')
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err):
            outcome = oauth.try_refresh_oauth_credentials(self._make_credentials())
        assert outcome.credentials is None
        assert outcome.error == "invalid_grant"

    def test_400_without_marker_is_transient(self):
        err = self._http_error(400, b'{"error": "temporarily_unavailable"}')
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err):
            outcome = oauth.try_refresh_oauth_credentials(self._make_credentials())
        assert outcome.error == "transient"

    def test_5xx_is_transient_even_with_marker(self):
        err = self._http_error(500, b'{"error": "invalid_grant"}')
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err):
            outcome = oauth.try_refresh_oauth_credentials(self._make_credentials())
        assert outcome.error == "transient"

    def test_network_error_is_transient(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            side_effect=urllib.error.URLError("dns"),
        ):
            outcome = oauth.try_refresh_oauth_credentials(self._make_credentials())
        assert outcome.error == "transient"

    def test_missing_refresh_token_is_permanent(self):
        creds = json.dumps({"claudeAiOauth": {"accessToken": "a", "expiresAt": 0}})
        outcome = oauth.try_refresh_oauth_credentials(creds)
        assert outcome.error == "no_refresh_token"

    def test_invalid_json_is_transient(self):
        # Changed contract (stale-credential robustness): an unparseable blob
        # is more likely a torn read than a credential shape — it must not
        # produce a permanent strike-advancing verdict.
        outcome = oauth.try_refresh_oauth_credentials("not json")
        assert outcome.error == "transient"

    def test_wrapper_returns_none_on_failure(self):
        err = self._http_error(400, b'{"error": "invalid_grant"}')
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err):
            assert oauth.refresh_oauth_credentials(self._make_credentials()) is None


class TestBuildTokenStatus:
    """Test token status formatting."""

    def test_builds_fresh_token_status(self):
        fixed_now = datetime(2026, 4, 2, 18, 0, 0, tzinfo=timezone.utc)
        expires_at = int(datetime(2026, 4, 2, 19, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
        credentials = json.dumps({
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
                "expiresAt": expires_at,
            }
        })

        with patch("claude_swap.oauth.datetime") as mock_dt:
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.fromtimestamp = datetime.fromtimestamp
            mock_dt.now.return_value = fixed_now
            status = oauth.build_token_status(credentials)

        assert status is not None
        assert "oauth: fresh, refresh token yes" in status
        assert "in 1h 30m" in status

    def test_builds_unknown_expiry_status(self):
        credentials = json.dumps({
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
            }
        })

        status = oauth.build_token_status(credentials)

        assert status == "oauth: unknown expiry, refresh token yes"


class TestFetchUsageForAccount:
    """Test refresh-aware usage fetches for managed accounts."""

    @staticmethod
    def _make_credentials(access="old-access", refresh="old-refresh",
                          expires_at=None, org_uuid="org-1", scopes=None):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if scopes is None:
            scopes = ["user:profile", "user:inference", "user:sessions:claude_code"]
        return json.dumps({
            "claudeAiOauth": {
                "accessToken": access,
                "refreshToken": refresh,
                "expiresAt": expires_at if expires_at is not None else now_ms + 3_600_000,
                "scopes": scopes,
                "subscriptionType": "pro",
                "rateLimitTier": "default_claude_ai",
            },
            "organizationUuid": org_uuid,
        })

    @staticmethod
    def _make_token_response(access="new-access", refresh="new-refresh",
                             expires_in=3600):
        return json.dumps({
            "access_token": access,
            "refresh_token": refresh,
            "expires_in": expires_in,
            "scope": "user:profile user:inference user:sessions:claude_code",
        }).encode()

    @staticmethod
    def _make_usage_response(h5_pct=12.0, d7_pct=34.0):
        resp = MagicMock()
        resp.read.return_value = json.dumps({
            "five_hour": {"utilization": h5_pct, "resets_at": None},
            "seven_day": {"utilization": d7_pct, "resets_at": None},
        }).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_refreshes_expired_token_before_usage_fetch(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        credentials = self._make_credentials(expires_at=now_ms - 1_000)

        token_resp = MagicMock()
        token_resp.read.return_value = self._make_token_response()
        token_resp.__enter__ = lambda s: s
        token_resp.__exit__ = MagicMock(return_value=False)

        usage_resp = self._make_usage_response()
        persist_mock = MagicMock()

        def mock_urlopen(req, timeout=0):
            if "oauth/token" in req.full_url:
                return token_resp
            if "oauth/usage" in req.full_url:
                assert req.get_header("Authorization") == "Bearer new-access"
                return usage_resp
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_usage_for_account(
                "1", "test@example.com", credentials,
                is_active=False,
                persist_credentials=persist_mock,
            )

        assert result is not None
        assert result["five_hour"]["pct"] == 12.0
        persist_mock.assert_called_once()
        persisted_creds = persist_mock.call_args[0][2]
        merged = json.loads(persisted_creds)
        assert merged["organizationUuid"] == "org-1"
        assert merged["claudeAiOauth"]["accessToken"] == "new-access"
        assert merged["claudeAiOauth"]["refreshToken"] == "new-refresh"

    def test_retries_401_with_token_refresh(self):
        """Account gets 401, refreshes, retries successfully."""
        credentials = self._make_credentials()

        token_resp = MagicMock()
        token_resp.read.return_value = self._make_token_response()
        token_resp.__enter__ = lambda s: s
        token_resp.__exit__ = MagicMock(return_value=False)

        usage_resp = self._make_usage_response(h5_pct=56.0, d7_pct=78.0)
        usage_calls = 0
        persist_mock = MagicMock()

        def mock_urlopen(req, timeout=0):
            nonlocal usage_calls
            if "oauth/token" in req.full_url:
                return token_resp
            if "oauth/usage" in req.full_url:
                usage_calls += 1
                if usage_calls == 1:
                    assert req.get_header("Authorization") == "Bearer old-access"
                    raise urllib.error.HTTPError(
                        req.full_url, 401, "Unauthorized", hdrs=None, fp=None,
                    )
                assert req.get_header("Authorization") == "Bearer new-access"
                return usage_resp
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_usage_for_account(
                "2", "test@example.com", credentials,
                is_active=False,
                persist_credentials=persist_mock,
            )

        assert result is not None
        assert result["seven_day"]["pct"] == 78.0
        assert usage_calls == 2
        persist_mock.assert_called_once()
        refreshed_oauth = json.loads(persist_mock.call_args[0][2])["claudeAiOauth"]
        assert refreshed_oauth["accessToken"] == "new-access"

    def test_valid_token_fetches_usage_without_refresh(self):
        """Account with valid token fetches usage without refresh."""
        credentials = self._make_credentials()

        usage_resp = self._make_usage_response(h5_pct=10.0, d7_pct=20.0)

        def mock_urlopen(req, timeout=0):
            if "oauth/usage" in req.full_url:
                assert req.get_header("Authorization") == "Bearer old-access"
                return usage_resp
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen), \
             patch("claude_swap.oauth.refresh_oauth_credentials") as refresh_mock:
            result = oauth.fetch_usage_for_account(
                "1", "test@example.com", credentials,
                is_active=False,
            )

        refresh_mock.assert_not_called()
        assert result is not None
        assert result["five_hour"]["pct"] == 10.0

    def test_refresh_failure_returns_none_gracefully(self):
        """If token refresh fails (e.g. revoked), usage returns None."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        credentials = self._make_credentials(expires_at=now_ms - 1_000)

        def mock_urlopen(req, timeout=0):
            if "oauth/token" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 400, "Bad Request", hdrs=None, fp=None,
                )
            if "oauth/usage" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 401, "Unauthorized", hdrs=None, fp=None,
                )
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_usage_for_account(
                "1", "test@example.com", credentials,
                is_active=False,
            )

        assert result is None

    def test_refreshes_when_scopes_are_missing(self):
        """Refresh should work even when stored credentials have no scopes."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        credentials = self._make_credentials(
            expires_at=now_ms - 1_000,
            scopes=None,
        )
        parsed = json.loads(credentials)
        del parsed["claudeAiOauth"]["scopes"]
        credentials = json.dumps(parsed)

        token_resp = MagicMock()
        token_resp.read.return_value = self._make_token_response()
        token_resp.__enter__ = lambda s: s
        token_resp.__exit__ = MagicMock(return_value=False)

        usage_resp = self._make_usage_response()
        persist_mock = MagicMock()

        def mock_urlopen(req, timeout=0):
            if "oauth/token" in req.full_url:
                body = json.loads(req.data.decode())
                assert "scope" not in body
                return token_resp
            if "oauth/usage" in req.full_url:
                return usage_resp
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_usage_for_account(
                "1", "test@example.com", credentials,
                is_active=False,
                persist_credentials=persist_mock,
            )

        assert result is not None
        persist_mock.assert_called_once()

    def test_active_account_skips_refresh_even_when_expired(self):
        """Active account with expired token must NOT trigger a refresh POST.

        Claude Code owns the active account's credentials and coordinates its
        own refresh via a lockfile on ~/.claude/ that cswap doesn't honor, so
        cswap must never touch the active account's tokens.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        credentials = self._make_credentials(expires_at=now_ms - 1_000)

        persist_mock = MagicMock()
        refresh_calls = 0

        def mock_urlopen(req, timeout=0):
            nonlocal refresh_calls
            if "oauth/token" in req.full_url:
                refresh_calls += 1
                raise AssertionError(
                    "Active account must not trigger a refresh POST"
                )
            if "oauth/usage" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 401, "Unauthorized", hdrs=None, fp=None,
                )
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_usage_for_account(
                "1", "test@example.com", credentials,
                is_active=True,
                persist_credentials=persist_mock,
            )

        assert refresh_calls == 0
        persist_mock.assert_not_called()
        # Usage call 401'd and there's no retry-with-refresh for active, so None.
        assert result is None

    def test_active_account_401_does_not_retry_with_refresh(self):
        """Active account that 401s returns None without attempting a refresh."""
        credentials = self._make_credentials()

        def mock_urlopen(req, timeout=0):
            if "oauth/token" in req.full_url:
                raise AssertionError(
                    "Active account must not trigger a refresh POST on 401"
                )
            if "oauth/usage" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 401, "Unauthorized", hdrs=None, fp=None,
                )
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        persist_mock = MagicMock()
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_usage_for_account(
                "1", "test@example.com", credentials,
                is_active=True,
                persist_credentials=persist_mock,
            )

        assert result is None
        persist_mock.assert_not_called()

    def test_persist_failure_logs_warning_with_recovery_hint(self, caplog, capsys):
        """If the persist callback raises, _persist logs at WARNING level with
        a recovery hint (re-run `cswap --add-account`), not debug, AND prints
        a user-visible warning to stderr, so a ``--json`` payload on stdout
        stays one parseable object.
        """
        import logging

        def boom(acct_num, acct_email, creds):
            raise RuntimeError("disk exploded")

        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            oauth._persist(boom, "1", "test@example.com", "{}")

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "claude-swap"
        ]
        assert len(warning_records) == 1
        msg = warning_records[0].getMessage()
        assert "failed to persist" in msg
        assert "cswap --add-account" in msg
        assert "1" in msg
        assert "test@example.com" in msg

        # Also verify the user-visible printed warning, and that stdout stays clean
        captured = capsys.readouterr()
        assert "failed to save refreshed token" in captured.err
        assert "cswap --add-account" in captured.err
        assert captured.out == ""


class TestClassifyUsageError:
    """Test _classify_usage_error kinds and Retry-After parsing."""

    @staticmethod
    def _http_error(code: int, headers: dict | None = None):
        import email.message
        hdrs = None
        if headers is not None:
            hdrs = email.message.Message()
            for k, v in headers.items():
                hdrs[k] = v
        return urllib.error.HTTPError(
            url="https://api.anthropic.com/api/oauth/usage",
            code=code, msg="err", hdrs=hdrs, fp=None,
        )

    def test_http_codes(self):
        assert oauth._classify_usage_error(self._http_error(429))[0] == "http-429"
        assert oauth._classify_usage_error(self._http_error(500))[0] == "http-500"
        assert oauth._classify_usage_error(self._http_error(401))[0] == "http-401"

    def test_retry_after_seconds(self):
        kind, retry = oauth._classify_usage_error(
            self._http_error(429, {"Retry-After": "30"})
        )
        assert kind == "http-429"
        assert retry == 30.0

    def test_retry_after_date_form_ignored(self):
        _, retry = oauth._classify_usage_error(
            self._http_error(429, {"Retry-After": "Fri, 04 Jul 2026 12:00:00 GMT"})
        )
        assert retry is None

    def test_retry_after_negative_clamped(self):
        _, retry = oauth._classify_usage_error(
            self._http_error(429, {"Retry-After": "-5"})
        )
        assert retry == 0.0

    def test_no_headers(self):
        kind, retry = oauth._classify_usage_error(self._http_error(429))
        assert (kind, retry) == ("http-429", None)

    def test_timeout(self):
        import socket
        assert oauth._classify_usage_error(TimeoutError())[0] == "timeout"
        assert oauth._classify_usage_error(socket.timeout())[0] == "timeout"
        assert oauth._classify_usage_error(
            urllib.error.URLError(TimeoutError())
        )[0] == "timeout"

    def test_network(self):
        assert oauth._classify_usage_error(
            urllib.error.URLError(ConnectionRefusedError())
        )[0] == "network"

    def test_bad_response(self):
        try:
            json.loads("not json")
        except json.JSONDecodeError as e:
            assert oauth._classify_usage_error(e)[0] == "bad-response"

    def test_fallback_type_name(self):
        assert oauth._classify_usage_error(ValueError("x"))[0] == "ValueError"


class TestTryFetchUsageOutcome:
    """Test try_fetch_usage_for_account outcome classification."""

    @staticmethod
    def _make_credentials() -> str:
        from datetime import timedelta
        future_ms = int(
            (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000
        )
        return json.dumps({
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
                "expiresAt": future_ms,
            }
        })

    def test_success_outcome(self):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"five_hour": {"utilization": 12.0, "resets_at": None}}
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=resp):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._make_credentials(), is_active=False,
            )
        assert outcome.error is None
        assert outcome.usage["five_hour"]["pct"] == 12.0

    def test_429_outcome_carries_retry_after(self, caplog):
        import email.message
        import logging
        hdrs = email.message.Message()
        hdrs["Retry-After"] = "42"
        err = urllib.error.HTTPError(
            "https://api.anthropic.com/api/oauth/usage", 429, "Too Many",
            hdrs=hdrs, fp=None,
        )
        with (
            patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err),
            caplog.at_level(logging.WARNING, logger="claude-swap"),
        ):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._make_credentials(), is_active=False,
            )
        assert outcome.usage is None
        assert outcome.error == "http-429"
        assert outcome.retry_after_s == 42.0
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        line = next(m for m in warnings if "http-429" in m)
        # The line users paste into public issues: account number and the
        # server's Retry-After, never the email.
        assert "account 1" in line
        assert "retry-after 42s" in line
        assert "a@b.c" not in line
        # Any 429 = the usage endpoint's own budget, which cumulative polling
        # across cswap surfaces can drain — the log says what is happening.
        # Deliberately not scoped to the token in the wording: the budget is
        # account/org-scoped (see poll_policy), so a re-login does not clear it.
        assert "usage-endpoint budget" in line

    def test_edge_429_warning_names_the_budget(self, caplog):
        import email.message
        import logging
        hdrs = email.message.Message()
        hdrs["Retry-After"] = "0"
        err = urllib.error.HTTPError(
            "https://api.anthropic.com/api/oauth/usage", 429, "Too Many",
            hdrs=hdrs, fp=None,
        )
        with (
            patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err),
            caplog.at_level(logging.WARNING, logger="claude-swap"),
        ):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._make_credentials(), is_active=False,
            )
        assert outcome.retry_after_s == 0.0
        line = next(
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "http-429" in r.getMessage()
        )
        # "Retry-After: 0" is the saturated-budget edge — same hint.
        assert "retry-after 0s" in line
        assert "usage-endpoint budget" in line

    def test_timeout_outcome(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            side_effect=urllib.error.URLError(TimeoutError()),
        ):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._make_credentials(), is_active=False,
            )
        assert outcome.error == "timeout"

    def test_no_access_token_outcome(self):
        outcome = oauth.try_fetch_usage_for_account(
            "1", "a@b.c", json.dumps({"claudeAiOauth": {}}), is_active=False,
        )
        assert outcome.error == "no-access-token"


class TestInvalidGrantPropagation:
    """A dead refresh-token lineage surfaces as error='invalid_grant', distinct
    from a transient 'refresh-failed', so the store can quarantine the account."""

    @staticmethod
    def _expired_credentials() -> str:
        from datetime import timedelta
        past_ms = int(
            (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000
        )
        return json.dumps({"claudeAiOauth": {
            "accessToken": "old-access", "refreshToken": "dead-refresh",
            "expiresAt": past_ms,
        }})

    @staticmethod
    def _valid_credentials() -> str:
        from datetime import timedelta
        future_ms = int(
            (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000
        )
        return json.dumps({"claudeAiOauth": {
            "accessToken": "good-access", "refreshToken": "dead-refresh",
            "expiresAt": future_ms,
        }})

    def test_proactive_refresh_invalid_grant_short_circuits(self):
        """Expired token + dead refresh: report invalid_grant without hitting usage."""
        with patch("claude_swap.oauth.try_refresh_oauth_credentials",
                   return_value=oauth.RefreshOutcome(None, "invalid_grant")), \
             patch("claude_swap.oauth.request_usage_data") as usage:
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._expired_credentials(), is_active=False,
            )
        assert outcome.error == "invalid_grant"
        usage.assert_not_called()  # no pointless 401/429 on a lost cause

    def test_401_retry_invalid_grant_is_permanent(self):
        """Valid-looking token, server 401, dead refresh → invalid_grant."""
        err = urllib.error.HTTPError(
            "https://api.anthropic.com/api/oauth/usage", 401, "Unauthorized",
            hdrs=None, fp=None,
        )
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err), \
             patch("claude_swap.oauth.try_refresh_oauth_credentials",
                   return_value=oauth.RefreshOutcome(None, "invalid_grant")):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._valid_credentials(), is_active=False,
            )
        assert outcome.error == "invalid_grant"

    def test_transient_refresh_failure_is_not_permanent(self):
        """A transient refresh failure stays 'refresh-failed', not invalid_grant."""
        err = urllib.error.HTTPError(
            "https://api.anthropic.com/api/oauth/usage", 401, "Unauthorized",
            hdrs=None, fp=None,
        )
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err), \
             patch("claude_swap.oauth.try_refresh_oauth_credentials",
                   return_value=oauth.RefreshOutcome(None, "transient")):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@b.c", self._valid_credentials(), is_active=False,
            )
        assert outcome.error == "refresh-failed"


class TestCredentialFingerprint:
    """Identity fingerprints for stored credentials (issue #117 guard)."""

    def test_stable_across_access_token_rotation(self):
        a = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-old", "refreshToken": "rt-1"}})
        b = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-new", "refreshToken": "rt-1", "expiresAt": 5}})
        assert oauth.credential_fingerprint(a) == oauth.credential_fingerprint(b)

    def test_differs_across_refresh_token_rotation(self):
        a = json.dumps({"claudeAiOauth": {"refreshToken": "rt-1"}})
        b = json.dumps({"claudeAiOauth": {"refreshToken": "rt-2"}})
        assert oauth.credential_fingerprint(a) != oauth.credential_fingerprint(b)

    def test_full_content_fallback_for_api_keys_and_setup_tokens(self):
        api_key = "sk-ant-api03-xyz"
        setup = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-abc"}})
        assert oauth.credential_fingerprint(api_key) is not None
        assert oauth.credential_fingerprint(setup) is not None
        # Never None for real bytes: a None would make every "did it change?"
        # comparison degenerate to "changed".
        assert oauth.credential_fingerprint(api_key) != oauth.credential_fingerprint(setup)

    def test_full_hash_never_collides_with_refresh_hash(self):
        with_rt = json.dumps({"claudeAiOauth": {"refreshToken": "rt-1"}})
        assert oauth.credential_fingerprint(with_rt).startswith("sha256:")
        assert oauth.credential_fingerprint("raw-token").startswith("sha256-full:")

    def test_empty_input_is_none(self):
        assert oauth.credential_fingerprint("") is None


class TestTokenAccountParsing:
    """The token endpoint's optional account identity must not be discarded."""

    _make_credentials = staticmethod(TestRefreshOAuthCredentials._make_credentials)

    def _refresh_with_response(self, payload: dict) -> oauth.RefreshOutcome:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(payload).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch(
            "claude_swap.oauth.urllib.request.urlopen", return_value=mock_response
        ):
            return oauth.try_refresh_oauth_credentials(self._make_credentials())

    def test_token_account_surfaced_when_present(self):
        outcome = self._refresh_with_response({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "account": {"uuid": "acc-uuid", "email_address": "a@b.c"},
            "organization": {"uuid": "org-uuid"},
        })
        assert outcome.error is None
        assert outcome.token_account == {
            "uuid": "acc-uuid", "email": "a@b.c", "organizationUuid": "org-uuid",
        }

    def test_token_account_absent_is_none(self):
        outcome = self._refresh_with_response({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        })
        assert outcome.error is None
        assert outcome.token_account is None

    # Same strict boundary as fetch_oauth_profile: identity is opportunistic
    # and must never break the refresh that carried it — malformed or
    # uuid-less data is None, optional fields normalize to str-or-None.

    def test_token_account_without_uuid_is_none(self):
        outcome = self._refresh_with_response({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "account": {"email_address": "a@b.c"},
        })
        assert outcome.error is None
        assert outcome.token_account is None

    def test_token_account_non_string_uuid_is_none(self):
        outcome = self._refresh_with_response({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "account": {"uuid": 12345, "email_address": "a@b.c"},
        })
        assert outcome.error is None
        assert outcome.token_account is None

    def test_token_account_uuid_whitespace_normalized(self):
        """Normalization happens at the boundary so padded uuids never reach
        comparisons or sequence.json backfills."""
        outcome = self._refresh_with_response({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "account": {"uuid": "  acc-uuid  ", "email_address": "a@b.c"},
        })
        assert outcome.token_account["uuid"] == "acc-uuid"

    def test_token_account_non_string_optionals_normalized(self):
        outcome = self._refresh_with_response({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
            "account": {"uuid": "acc-uuid", "email_address": {"weird": 1}},
            "organization": {"uuid": 99},
        })
        assert outcome.error is None
        assert outcome.token_account == {
            "uuid": "acc-uuid", "email": None, "organizationUuid": None,
        }


@pytest.mark.no_oauth_profile_fake
class TestFetchOauthProfile:
    """Access-token → account-identity resolution (/api/oauth/profile)."""

    def _profile_response(self, payload: dict):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(payload).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    def test_resolves_identity(self):
        seen = {}

        def mock_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["auth"] = req.headers.get("Authorization")
            return self._profile_response({
                "account": {"uuid": "acc-uuid", "email": "a@b.c"},
                "organization": {"uuid": "org-uuid"},
            })

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            result = oauth.fetch_oauth_profile("sk-live")
        assert result == {
            "uuid": "acc-uuid", "email": "a@b.c", "organizationUuid": "org-uuid",
        }
        assert seen["url"].endswith("/api/oauth/profile")
        assert seen["auth"] == "Bearer sk-live"

    def test_uses_bounded_timeout(self):
        """One bounded call: the profile lookup may only ever add latency,
        never hang a switch."""
        seen = {}

        def mock_urlopen(req, timeout=0):
            seen["timeout"] = timeout
            return self._profile_response({
                "account": {"uuid": "acc-uuid", "email": "a@b.c"},
            })

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=mock_urlopen):
            oauth.fetch_oauth_profile("sk-live")
        assert seen["timeout"] == 5

    def test_network_failure_is_unresolvable_not_error(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            assert oauth.fetch_oauth_profile("sk-live") is None

    def test_missing_account_object_is_unresolvable(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({"unexpected": True}),
        ):
            assert oauth.fetch_oauth_profile("sk-live") is None

    # Strict resolution boundary: the oracle is advisory (None keeps the
    # switch on the fail-open path), so a response only counts as resolved
    # with a non-empty string account.uuid — a schema change must degrade to
    # pre-fix behavior, not to preserve-and-skip.

    def test_missing_uuid_is_unresolvable(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({
                "account": {"email": "a@b.c"},
                "organization": {"uuid": "org-uuid"},
            }),
        ):
            assert oauth.fetch_oauth_profile("sk-live") is None

    def test_non_string_uuid_is_unresolvable(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({
                "account": {"uuid": 12345, "email": "a@b.c"},
            }),
        ):
            assert oauth.fetch_oauth_profile("sk-live") is None

    def test_blank_uuid_is_unresolvable(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({
                "account": {"uuid": "   ", "email": "a@b.c"},
            }),
        ):
            assert oauth.fetch_oauth_profile("sk-live") is None

    def test_malformed_json_is_unresolvable(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b"<!doctype html><html>gateway error"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch(
            "claude_swap.oauth.urllib.request.urlopen", return_value=mock_response,
        ):
            assert oauth.fetch_oauth_profile("sk-live") is None

    def test_uuid_whitespace_normalized_at_boundary(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({
                "account": {"uuid": "  acc-uuid  ", "email": "a@b.c"},
            }),
        ):
            result = oauth.fetch_oauth_profile("sk-live")
        assert result["uuid"] == "acc-uuid"

    def test_valid_uuid_with_missing_email_still_resolves(self):
        """email/organization are optional; uuid is the identity."""
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({
                "account": {"uuid": "acc-uuid"},
            }),
        ):
            result = oauth.fetch_oauth_profile("sk-live")
        assert result == {"uuid": "acc-uuid", "email": None, "organizationUuid": None}

    def test_non_string_optional_fields_are_dropped_not_fatal(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            return_value=self._profile_response({
                "account": {"uuid": "acc-uuid", "email": {"weird": True}},
                "organization": {"uuid": 99},
            }),
        ):
            result = oauth.fetch_oauth_profile("sk-live")
        assert result == {"uuid": "acc-uuid", "email": None, "organizationUuid": None}

    def test_401_is_unresolvable_with_log_file_warning(self, caplog):
        """401 is evidence (the live token can't authenticate) but not proof —
        fail open, and record it at warning level in the log only (the
        console handler exists only under --debug)."""
        import logging

        err = urllib.error.HTTPError(
            "https://api.anthropic.com/api/oauth/profile", 401,
            "Unauthorized", {}, None,
        )
        with patch(
            "claude_swap.oauth.urllib.request.urlopen", side_effect=err,
        ), caplog.at_level(logging.WARNING, logger="claude-swap"):
            assert oauth.fetch_oauth_profile("sk-live") is None
        assert any(
            "401" in r.message and "pre-fix" in r.message
            for r in caplog.records
        )


class TestInvalidGrantTaxonomy:
    """M3: the permanent invalid_grant verdict requires an RFC 6749 §5.2
    parse — top-level error == "invalid_grant" in the JSON body. Substring
    hits inside other envelopes stay transient; invalid_client is a distinct
    systemic kind, never a dead-token verdict."""

    def _refresh_with_body(self, monkeypatch, code, body):
        import urllib.error, io
        creds = json.dumps({
            "claudeAiOauth": {"refreshToken": "rt-x", "accessToken": "a"}
        })

        def raise_http(*a, **k):
            raise urllib.error.HTTPError(
                "url", code, "err", {}, io.BytesIO(body.encode())
            )

        monkeypatch.setattr(
            "claude_swap.oauth.urllib.request.urlopen", raise_http
        )
        return oauth.try_refresh_oauth_credentials(creds)

    def test_rfc_invalid_grant_is_permanent(self, monkeypatch):
        out = self._refresh_with_body(
            monkeypatch, 400, '{"error": "invalid_grant"}'
        )
        assert out.error == "invalid_grant"

    def test_substring_in_other_envelope_is_transient(self, monkeypatch):
        # the marker appears only inside a nested message — not a §5.2 error
        out = self._refresh_with_body(
            monkeypatch, 400,
            '{"error": "server_error", "detail": "log mentions invalid_grant"}'
        )
        assert out.error == "transient"

    def test_invalid_client_is_systemic_not_dead_token(self, monkeypatch):
        out = self._refresh_with_body(
            monkeypatch, 401, '{"error": "invalid_client"}'
        )
        assert out.error == "invalid_client"

    def test_unparseable_body_is_transient(self, monkeypatch):
        out = self._refresh_with_body(monkeypatch, 400, "<html>oops</html>")
        assert out.error == "transient"

    def test_error_description_variant_still_permanent(self, monkeypatch):
        out = self._refresh_with_body(
            monkeypatch, 400,
            '{"error": "invalid_grant", "error_description": "revoked"}'
        )
        assert out.error == "invalid_grant"


class TestNoRefreshTokenStructuralGuard:
    """M3: ``no_refresh_token`` is permanent only for a structurally complete
    OAuth dict genuinely missing the field — an unparseable/partial blob is
    transient (a torn read must not condemn the slot)."""

    def test_complete_dict_without_rt_is_permanent(self):
        creds = json.dumps({"claudeAiOauth": {"accessToken": "a"}})
        out = oauth.try_refresh_oauth_credentials(creds)
        assert out.error == "no_refresh_token"

    def test_unparseable_blob_is_transient(self):
        out = oauth.try_refresh_oauth_credentials('{"claudeAiOa')  # torn read
        assert out.error == "transient"

    def test_non_dict_payload_is_transient(self):
        out = oauth.try_refresh_oauth_credentials('"just-a-string"')
        assert out.error == "transient"


class TestConsumeBusyIsDeterministic:
    """A busy consume gate must not fall through to a guaranteed 401.

    `consume-busy` means another process holds the gate — the token in hand is
    known-expired, so calling the usage endpoint with it 401s every time, and
    the retry re-enters the gate and gets busy again. The kind then arrives as
    generic "refresh-failed", hiding the distinct kind this PR added and
    spending a request per pass to learn nothing.
    """

    def test_a_busy_gate_does_not_spend_a_doomed_request(self):
        creds = json.dumps({
            "claudeAiOauth": {
                "accessToken": "expired",
                "refreshToken": "r",
                "expiresAt": 1,  # long past
            }
        })
        with patch("claude_swap.oauth.request_usage_data") as usage:
            out = oauth.try_fetch_usage_for_account(
                "1", "a@example.com", creds, is_active=False,
                refresh_via=lambda *_: oauth.RefreshOutcome(None, "consume-busy"),
            )
        assert out.error == "consume-busy", out.error
        usage.assert_not_called()

    def test_every_deterministic_kind_has_a_note(self):
        """The reason these kinds stay distinct is the note they carry.

        ``try_fetch_usage_for_account`` keeps a deterministic kind rather than
        collapsing it to "refresh-failed" because "ERROR_NOTES renders the
        remedy for each" — a kind with no note renders the bare identifier,
        which is strictly worse than the generic string it displaced.
        """
        from claude_swap.switcher import ERROR_NOTES

        missing = [
            k for k in oauth._DETERMINISTIC_REFRESH_ERRORS if k not in ERROR_NOTES
        ]
        assert not missing, missing


class TestLoginExpiresAtIso:
    def test_refresh_token_expiry_is_reported_as_iso_utc(self):
        creds = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-x", "refreshTokenExpiresAt": 1791421596865,
        }})
        assert oauth.login_expires_at_iso(creds) == "2026-10-08T01:06:36Z"

    @pytest.mark.parametrize("creds", [
        "",
        "not json",
        json.dumps({"claudeAiOauth": {"accessToken": "sk-x"}}),
        json.dumps({"claudeAiOauth": {"refreshTokenExpiresAt": "soon"}}),
        json.dumps({"claudeAiOauth": {"refreshTokenExpiresAt": True}}),
        json.dumps({"claudeAiOauth": {"refreshTokenExpiresAt": 0}}),
        json.dumps({"other": {}}),
    ])
    def test_anything_but_a_positive_epoch_is_unknown(self, creds):
        assert oauth.login_expires_at_iso(creds) is None


# --- Dollar-budget (Enterprise) accounts -------------------------------------
#
# Both fixtures below are trimmed transcriptions of live
# `GET /api/oauth/usage` responses captured 2026-09-15: one Enterprise account
# gated by a dollar budget, one ordinary subscription account. Keep them shaped
# like the wire format, null slots and all — the null-vs-absent distinction and
# the `monthly_limit: 0` on the ordinary account are precisely what the parsing
# has to get right.

BUDGET_ACCOUNT_RESPONSE = {
    "five_hour": None,
    "seven_day": None,
    "seven_day_opus": None,
    "seven_day_sonnet": None,
    "seven_day_cowork": None,
    "seven_day_omelette": None,
    "seven_day_oauth_apps": None,
    "tangelo": None,
    "iguana_necktie": None,
    "omelette_promotional": None,
    # Non-null on BOTH captured accounts, and carrying nothing but a zero
    # utilization: presence is not an Enterprise marker, and this must never
    # produce a row.
    "nimbus_quill": {
        "utilization": 0.0, "resets_at": None,
        "limit_dollars": None, "used_dollars": None, "remaining_dollars": None,
        "locked_reason": None,
    },
    # The plan's included dollar pool: spent, yet the account kept serving
    # requests because usage falls through to the credit pool below.
    "cinder_cove": {
        "utilization": 100.0,
        "resets_at": "2026-09-18T02:10:41.779260+00:00",
        "limit_dollars": 1000, "used_dollars": 1000.0, "remaining_dollars": 0.0,
        "locked_reason": None,
    },
    "copper_kite": None, "harbor_lantern": None, "amber_ladder": None,
    "juniper_tide": None, "cedar_ember": None,
    "extra_usage": {
        "is_enabled": True, "monthly_limit": 20000, "used_credits": 6123.0,
        "utilization": 30.615, "currency": "USD", "decimal_places": 2,
        "disabled_reason": None, "user_disabled": False,
        "spend_limit_reached": False, "credits_ever_enabled": True,
        "daily": None, "weekly": None,
    },
    "limits": [],
    "spend": {
        "used": {"amount_minor": 6123, "currency": "USD", "exponent": 2},
        "limit": {"amount_minor": 20000, "currency": "USD", "exponent": 2},
        "percent": 31, "severity": "normal", "enabled": True,
        "disabled_reason": None,
        "cap": {"money": None, "credits": {"amount_minor": 20000, "exponent": 2}},
        "balance": None, "auto_reload": None,
        "can_purchase_credits": False, "can_toggle": False,
    },
    "member_dashboard_available": True,
    "seven_day_breakdown": None,
}

WINDOW_ACCOUNT_RESPONSE = {
    "five_hour": {"utilization": 0.0, "resets_at": None, "limit_dollars": None,
                  "used_dollars": None, "remaining_dollars": None,
                  "locked_reason": None},
    "seven_day": {"utilization": 0.0, "resets_at": "2026-09-21T16:00:00+00:00",
                  "limit_dollars": None, "used_dollars": None,
                  "remaining_dollars": None, "locked_reason": None},
    "nimbus_quill": {"utilization": 0.0, "resets_at": None,
                     "limit_dollars": None, "used_dollars": None,
                     "remaining_dollars": None, "locked_reason": None},
    "cinder_cove": None,
    # The trap: extra usage is *enabled* with a limit of 0. That is "no credit
    # pool", not "a $0 pool that is spent".
    "extra_usage": {"is_enabled": True, "monthly_limit": 0, "used_credits": 0.0,
                    "utilization": None, "currency": "USD", "decimal_places": 2,
                    "disabled_reason": None, "user_disabled": False},
    "limits": [
        {"kind": "session", "group": "session", "percent": 0,
         "resets_at": None, "scope": None, "is_active": True},
        {"kind": "weekly_all", "group": "weekly", "percent": 0,
         "resets_at": "2026-09-21T16:00:00+00:00", "scope": None,
         "is_active": False},
        {"kind": "weekly_scoped", "group": "weekly", "percent": 0,
         "resets_at": "2026-09-21T16:00:00+00:00",
         "scope": {"model": {"id": None, "display_name": "Fable"},
                   "surface": None},
         "is_active": False},
    ],
    "spend": {"used": {"amount_minor": 0, "currency": "USD", "exponent": 2},
              "limit": {"amount_minor": 0, "currency": "USD", "exponent": 2},
              "percent": 0, "severity": "normal", "enabled": True,
              "cap": {"money": None,
                      "credits": {"amount_minor": 0, "exponent": 2}}},
    "member_dashboard_available": False,
}


def _without(response: dict, *keys: str) -> dict:
    return {k: v for k, v in response.items() if k not in keys}


class TestBudgetWindowParsing:
    """build_usage_result's generic recognition of dollar pools."""

    def test_dollar_pool_becomes_a_budget_window(self):
        result = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)
        assert len(result["budget"]) == 1
        plan = result["budget"][0]
        assert plan["name"] == "plan"
        assert plan["pct"] == 100.0
        assert plan["used"] == 1000.0
        assert plan["limit"] == 1000.0
        assert plan["resets_at"] == "2026-09-18T02:10:41.779260+00:00"
        assert "countdown" in plan and "clock" in plan

    def test_code_name_is_carried_but_is_never_the_label(self):
        # The key rotates and means nothing to a user, so it may ride along for
        # JSON/debug but the display name must be positional.
        plan = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)["budget"][0]
        assert plan["key"] == "cinder_cove"
        assert plan["name"] == "plan"

    def test_null_dollar_fields_are_not_a_budget_window(self):
        # `nimbus_quill` is non-null on the budget account and still must not
        # produce a row: only `limit_dollars` discriminates.
        keys = [w["key"] for w in oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)["budget"]]
        assert keys == ["cinder_cove"]

    def test_window_account_reports_no_budget_at_all(self):
        # Same `nimbus_quill` object arrives on an ordinary account (measured
        # 2026-09-15) — nothing there is a budget.
        result = oauth.build_usage_result(WINDOW_ACCOUNT_RESPONSE)
        assert "budget" not in result
        assert result["five_hour"]["pct"] == 0.0
        assert result["seven_day"]["pct"] == 0.0
        assert [w["name"] for w in result["scoped"]] == ["Fable"]

    def test_a_never_seen_code_name_parses_generically(self):
        # Nothing may be keyed on the current names: they rotate.
        result = oauth.build_usage_result({
            "velvet_harbor": {"utilization": 42.5, "resets_at": None,
                              "limit_dollars": 500, "used_dollars": 212.5,
                              "remaining_dollars": 287.5, "locked_reason": None},
        })
        assert result["budget"] == [
            {"key": "velvet_harbor", "name": "plan", "pct": 42.5,
             "used": 212.5, "limit": 500.0},
        ]

    def test_pools_are_numbered_in_api_order(self):
        result = oauth.build_usage_result({
            "cinder_cove": {"utilization": 100.0, "limit_dollars": 1000,
                            "used_dollars": 1000.0},
            "juniper_tide": {"utilization": 10.0, "limit_dollars": 250,
                             "used_dollars": 25.0},
            "amber_ladder": {"utilization": 0.0, "limit_dollars": 50,
                             "used_dollars": None},
        })
        assert [(w["name"], w["limit"]) for w in result["budget"]] == [
            ("plan", 1000.0), ("plan 2", 250.0), ("plan 3", 50.0),
        ]
        # A null `used_dollars` drops just that field, like every other window.
        assert "used" not in result["budget"][2]

    @pytest.mark.parametrize("pool", [
        {"utilization": 50.0, "limit_dollars": 0},      # no pool, not a spent one
        {"utilization": 50.0, "limit_dollars": -5},
        {"utilization": 50.0, "limit_dollars": None},
        {"utilization": 50.0, "limit_dollars": "1000"},
        {"utilization": 50.0, "limit_dollars": True},   # bool is not a number
        {"utilization": None, "limit_dollars": 1000},
        {"utilization": True, "limit_dollars": 1000},
        {"utilization": "100", "limit_dollars": 1000},
    ])
    def test_only_a_real_positive_limit_makes_a_budget(self, pool):
        assert oauth.build_usage_result({"cinder_cove": pool}) is None

    def test_unparseable_reset_costs_only_the_clock_strings(self):
        result = oauth.build_usage_result({
            "cinder_cove": {"utilization": 7.0, "resets_at": "whenever",
                            "limit_dollars": 100, "used_dollars": 7.0},
        })
        plan = result["budget"][0]
        assert plan["pct"] == 7.0
        assert plan["resets_at"] == "whenever"
        assert "countdown" not in plan and "clock" not in plan

    def test_skipped_non_null_pools_are_logged_once(self, caplog):
        # The discoverability valve for the next rotation of these names.
        import logging
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="claude-swap"):
            oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)
        lines = [
            r.getMessage() for r in caplog.records
            if "not read as budgets" in r.getMessage()
        ]
        assert len(lines) == 1
        assert "nimbus_quill" in lines[0]
        assert "cinder_cove" not in lines[0]   # it WAS read as a budget
        assert "juniper_tide" not in lines[0]  # null slot: never a candidate
        assert "extra_usage" not in lines[0]   # handled elsewhere

    def test_a_response_of_only_noise_is_still_nothing(self):
        assert oauth.build_usage_result({
            "nimbus_quill": BUDGET_ACCOUNT_RESPONSE["nimbus_quill"],
        }) is None


class TestSpendGuards:
    """The zero-limit trap and the top-level `spend` fallback."""

    def test_zero_monthly_limit_is_no_pool_not_an_empty_one(self):
        # THE regression to avoid: a 0/0 spend row reads as 100% used, and
        # money can now bind a decision.
        assert "spend" not in oauth.build_usage_result(WINDOW_ACCOUNT_RESPONSE)

    def test_zero_limit_is_rejected_even_when_utilization_is_present(self):
        result = oauth.build_usage_result({
            "five_hour": {"utilization": 3.0},
            "extra_usage": {"is_enabled": True, "monthly_limit": 0,
                            "used_credits": 0.0, "utilization": 0.0},
        })
        assert "spend" not in result
        assert result["five_hour"]["pct"] == 3.0

    def test_extra_usage_wins_over_the_top_level_spend_object(self):
        # Both describe the same pool; extra_usage carries the finer number
        # (30.615 against the rounded 31), so it must stay the source.
        spend = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)["spend"]
        assert spend["pct"] == 30.615
        assert spend["used"] == 61.23
        assert spend["limit"] == 200.0
        assert spend["currency"] == "USD"

    def test_top_level_spend_is_used_when_extra_usage_yields_nothing(self):
        result = oauth.build_usage_result(
            {**BUDGET_ACCOUNT_RESPONSE, "extra_usage": None}
        )
        assert result["spend"] == {
            "used": 61.23, "limit": 200.0, "pct": 31.0, "currency": "USD",
        }

    def test_top_level_spend_honours_the_same_zero_guard(self):
        result = oauth.build_usage_result(
            {**WINDOW_ACCOUNT_RESPONSE, "extra_usage": None}
        )
        assert "spend" not in result

    def test_top_level_spend_scales_by_its_own_exponent(self):
        result = oauth.build_usage_result({
            "spend": {"used": {"amount_minor": 1500, "currency": "EUR",
                               "exponent": 3},
                      "limit": {"amount_minor": 90000, "currency": "EUR",
                                "exponent": 3},
                      "percent": 1.67},
        })
        assert result["spend"]["used"] == 1.5
        assert result["spend"]["limit"] == 90.0
        assert result["spend"]["currency"] == "EUR"

    @pytest.mark.parametrize("sp", [
        {"used": {"amount_minor": 1}, "limit": {"amount_minor": 100}},  # no percent
        {"used": {"amount_minor": 1}, "percent": 1},                    # no limit
        {"limit": {"amount_minor": 100}, "percent": 1},                 # no used
        {"used": {"amount_minor": 1}, "limit": {"amount_minor": "100"},
         "percent": 1},
        {"used": {"amount_minor": 1}, "limit": {"amount_minor": 100},
         "percent": True},
    ])
    def test_malformed_top_level_spend_yields_nothing(self, sp):
        assert oauth.build_usage_result({"spend": sp}) is None


class TestBudgetAccountClassification:
    """has_rate_windows / is_budget_account / budget_windows."""

    def test_captured_shapes(self):
        budget = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)
        window = oauth.build_usage_result(WINDOW_ACCOUNT_RESPONSE)
        assert oauth.has_rate_windows(budget) is False
        assert oauth.is_budget_account(budget) is True
        assert oauth.has_rate_windows(window) is True
        assert oauth.is_budget_account(window) is False

    @pytest.mark.parametrize("usage", [None, {}, "api key", [], {"budget": []}])
    def test_unknown_usage_is_not_a_budget_account(self, usage):
        # "We could not measure this account" must never read as "budget".
        assert oauth.is_budget_account(usage) is False
        assert oauth.has_rate_windows(usage) is False
        assert oauth.budget_windows(usage) == []

    def test_scoped_only_account_with_credits_is_not_a_budget_account(self):
        # has_rate_windows is deliberately independent of the `models` filter:
        # a Max account whose config names no models still has rate windows,
        # and its credits must stay a separate axis.
        usage = {"scoped": [{"name": "Fable", "pct": 10.0}],
                 "spend": {"pct": 99.0}}
        assert oauth.has_rate_windows(usage) is True
        assert oauth.is_budget_account(usage) is False
        assert oauth.relevant_windows(usage) == []

    def test_budget_alone_without_credits_is_a_budget_account(self):
        usage = {"budget": [{"key": "cinder_cove", "name": "plan", "pct": 12.0,
                             "limit": 1000.0}]}
        assert oauth.is_budget_account(usage) is True
        assert oauth.budget_windows(usage) == usage["budget"]

    def test_budget_windows_filters_unusable_rows(self):
        # A persisted `last_good` from an older/garbled row must not hand
        # consumers something they have to re-validate.
        usage = {"budget": [
            "not a dict",
            {"name": "plan", "pct": None},
            {"pct": 10.0},
            {"name": "plan", "pct": True},
            {"name": "plan 2", "pct": 10.0},
        ]}
        assert oauth.budget_windows(usage) == [{"name": "plan 2", "pct": 10.0}]

    def test_malformed_five_hour_does_not_count_as_a_rate_window(self):
        assert oauth.has_rate_windows({"five_hour": {"pct": None}}) is False


class TestBudgetRelevantWindows:
    """relevant_windows / account_headroom for money-gated accounts."""

    def test_credits_bind_not_the_exhausted_plan_pool(self):
        # Measured 2026-09-15: the plan pool was at 100% while the account kept
        # serving requests and credits kept climbing. max() over the pools
        # would call this healthy account exhausted.
        usage = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)
        assert usage["budget"][0]["pct"] == 100.0
        assert oauth.relevant_windows(usage) == [("$$", 30.615, None)]
        assert oauth.account_headroom(usage) == pytest.approx(69.385)

    def test_models_filter_does_not_disturb_the_budget_answer(self):
        usage = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)
        assert oauth.account_headroom(usage, ["Fable"]) == pytest.approx(69.385)
        assert oauth.account_headroom(usage, ["all"]) == pytest.approx(69.385)

    def test_exhausted_credits_are_at_limit(self):
        usage = oauth.build_usage_result({
            **BUDGET_ACCOUNT_RESPONSE,
            "extra_usage": {**BUDGET_ACCOUNT_RESPONSE["extra_usage"],
                            "used_credits": 20000.0, "utilization": 100.0},
        })
        assert oauth.account_headroom(usage) == 0.0

    def test_plan_pool_binds_when_there_is_no_credit_pool(self):
        usage = oauth.build_usage_result(
            _without({**BUDGET_ACCOUNT_RESPONSE, "extra_usage": None}, "spend")
        )
        assert oauth.relevant_windows(usage) == [
            ("plan", 100.0, "2026-09-18T02:10:41.779260+00:00"),
        ]
        assert oauth.account_headroom(usage) == 0.0

    def test_every_plan_pool_is_listed_when_there_are_no_credits(self):
        usage = {"budget": [
            {"name": "plan", "pct": 100.0, "resets_at": "2026-09-18T02:10:41+00:00"},
            {"name": "plan 2", "pct": 40.0},
        ]}
        assert oauth.relevant_windows(usage) == [
            ("plan", 100.0, "2026-09-18T02:10:41+00:00"),
            ("plan 2", 40.0, None),
        ]
        assert oauth.account_headroom(usage) == 0.0

    def test_labels_stay_human_readable(self):
        # These reach user-facing strings ("at X limit"), so a rotating API
        # code name must never be one of them.
        usage = oauth.build_usage_result(BUDGET_ACCOUNT_RESPONSE)
        assert [label for label, _, _ in oauth.relevant_windows(usage)] == ["$$"]

    def test_window_account_is_untouched(self):
        usage = oauth.build_usage_result(WINDOW_ACCOUNT_RESPONSE)
        assert oauth.relevant_windows(usage) == [
            ("5h", 0.0, None),
            ("7d", 0.0, "2026-09-21T16:00:00+00:00"),
        ]
        assert oauth.account_headroom(usage) == 100.0

    def test_a_budget_account_with_neither_pool_is_still_unknown(self):
        assert oauth.relevant_windows({"five_hour": {"pct": None}}) == []
        assert oauth.account_headroom({"five_hour": {"pct": None}}) is None
