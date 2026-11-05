"""The claude-swap Textual application.

Owns the snapshot poll loop and every mutating action (switch/add/remove),
so the dashboard, the auto view, and the command palette all drive the same
code paths. Blocking switcher work always runs in thread workers — the UI
loop never touches file locks, keychain subprocesses, or the network.
"""

from __future__ import annotations

import time
from dataclasses import replace
from functools import partial
from typing import Callable

from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive
from textual.worker import WorkerState

from claude_swap import printer
from claude_swap.models import AccountsSnapshot
from claude_swap.snapshot_source import account_identity
from claude_swap.settings import load_settings, load_ui_settings, set_setting
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.tui.autoview import AutoScreen
from claude_swap.tui.dashboard import DashboardScreen, WatchScreen
from claude_swap.tui.data import ActionResult, SnapshotSource, format_duration, run_action
from claude_swap.tui.modals import (
    AddTokenModal,
    ConfirmModal,
    IdentityForm,
    IdentityModal,
    OutputModal,
    TokenForm,
)
from claude_swap.tui.theme import CSWAP_DARK, CSWAP_LIGHT


class CswapApp(App):
    """claude-swap interactive dashboard."""

    TITLE = "claude-swap"
    CSS_PATH = "cswap.tcss"
    # No command palette: actions live in the dashboard's nested menu, in
    # their own context — not in a global searchable list.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("ctrl+t", "toggle_theme", "Theme")]

    POLL_INTERVAL_S = 3.0  # matches the old watch view's recapture cadence
    # Snapshot age stays hidden while polling is healthy (age never exceeds
    # ~POLL_INTERVAL_S); past this it surfaces as a staleness alarm. 60s also
    # keeps format_duration in whole minutes, so the note never ticks per
    # second.
    SNAPSHOT_AGE_NOTE_S = 60.0

    snapshot: reactive[AccountsSnapshot | None] = reactive(None)
    refresh_status: reactive[str] = reactive("")
    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        switcher: ClaudeAccountSwitcher,
        *,
        start: str = "dashboard",
        detected: str | None = None,
    ) -> None:
        super().__init__()
        self.switcher = switcher
        self._start = start  # "dashboard" | "watch" (`cswap watch`)
        self._detected = detected  # terminal background sensed pre-driver, or None
        self.source = SnapshotSource(switcher)
        self._store_only = False
        self._full_next = False
        self._normal_refreshing = False
        self._store_refreshing = False
        self._normal_started_at: float | None = None
        self._refresh_generation = 0
        self._applied_generation = 0
        self._last_refresh_error = ""
        # Completion callback of the one in-flight action, if it wants one.
        self._action_then: Callable[[bool], None] | None = None
        # The auto-switch lines, drawn as ticks on the status strip's bars
        # everywhere: the threshold on the 5h row, the weekly line (when
        # set) on the 7d and per-model rows. Missing/invalid settings fall
        # back to the default.
        try:
            loaded = load_settings(switcher.backup_dir)
            self.threshold_pct: float | None = loaded.threshold
            self.weekly_threshold_pct: float | None = loaded.weekly_threshold
        except Exception:
            self.threshold_pct = None
            self.weekly_threshold_pct = None
        try:
            self._theme_name = load_ui_settings(switcher.backup_dir).theme
        except Exception:
            self._theme_name = "auto"

    def on_mount(self) -> None:
        self.register_theme(CSWAP_DARK)
        self.register_theme(CSWAP_LIGHT)
        resolved = self._resolved_theme()
        # We own the theme; $TEXTUAL_THEME is intentionally not honoured.
        self.theme = f"cswap-{resolved}"
        printer.set_theme(resolved)
        self.push_screen(DashboardScreen())
        if self._start == "watch":
            # Stacked over the dashboard so Esc lands there, not on exit.
            self.push_screen(WatchScreen())
        self.set_interval(self.POLL_INTERVAL_S, self._tick)
        self.set_interval(1.0, self._update_refresh_status)
        self._tick()

    # -- snapshot poll loop ---------------------------------------------------

    def _tick(self) -> None:
        """Start one eligible refresh lane.

        Normal mode prefers the fetch-enabled lane. When that lane is blocked,
        the poll tick may still observe another process's store update through
        one store-only lane. Auto mode already has an engine fetching, so it
        launches only store-only snapshots.
        """
        if self._store_only:
            self._start_store_refresh()
        elif not self._normal_refreshing:
            full, self._full_next = self._full_next, False
            self._start_normal_refresh(full=full)
        else:
            self._start_store_refresh()

    def _start_normal_refresh(self, *, full: bool) -> None:
        if self._normal_refreshing:
            return
        self._normal_refreshing = True
        self._normal_started_at = time.time()
        generation = self._next_refresh_generation()
        self._update_refresh_status()
        self.run_worker(
            partial(self._refresh_blocking, generation, "normal", full, False),
            thread=True,
            group="refresh-normal",
            exit_on_error=False,
            name="snapshot-refresh",
        )

    def _start_store_refresh(self) -> None:
        if self._store_refreshing:
            return
        self._store_refreshing = True
        generation = self._next_refresh_generation()
        self._update_refresh_status()
        self.run_worker(
            partial(self._refresh_blocking, generation, "store", False, True),
            thread=True,
            group="refresh-store",
            exit_on_error=False,
            name="snapshot-store-refresh",
        )

    def _next_refresh_generation(self) -> int:
        self._refresh_generation += 1
        return self._refresh_generation

    def _refresh_blocking(
        self, generation: int, lane: str, full: bool, store_only: bool
    ) -> None:
        snap = self.source.take(full=full, store_only=store_only)
        self.call_from_thread(self._apply_snapshot, generation, lane, snap)
        if self._store_only:
            self.source.refresh_plans()

    def _apply_snapshot(
        self, generation: int, lane: str, snap: AccountsSnapshot
    ) -> None:
        if lane == "normal":
            self._normal_refreshing = False
            self._normal_started_at = None
        else:
            self._store_refreshing = False
        self._last_refresh_error = ""
        if generation >= self._applied_generation:
            self._applied_generation = generation
            self.snapshot = snap
        elif self.snapshot is not None:
            # A later-started store repaint owns account metadata, but the
            # older worker may have completed a genuinely newer provider fetch.
            # SnapshotSource has already rejected per-account regressions, so
            # merge its canonical usage rows without restoring stale metadata.
            current = self.snapshot
            incoming = {acc.number: acc for acc in snap.accounts}
            accounts = tuple(
                replace(acc, usage=other.usage)
                if (
                    (other := incoming.get(acc.number)) is not None
                    and account_identity(acc) == account_identity(other)
                )
                else acc
                for acc in current.accounts
            )
            self.snapshot = replace(
                current,
                accounts=accounts,
                taken_at=max(current.taken_at, snap.taken_at),
            )
        self._update_refresh_status()

    def _update_refresh_status(self) -> None:
        parts: list[str] = []
        now = time.time()
        if self.snapshot is not None:
            age = max(0.0, now - self.snapshot.taken_at)
            if age >= self.SNAPSHOT_AGE_NOTE_S:
                parts.append(f"snapshot {format_duration(age)} ago")
        if self._normal_refreshing and self._normal_started_at is not None:
            elapsed = now - self._normal_started_at
            if elapsed >= self.POLL_INTERVAL_S:
                parts.append(f"refreshing {format_duration(elapsed)}")
        self.refresh_status = " · ".join(parts)

    def request_refresh(self, *, full: bool = False) -> None:
        if full:
            self._full_next = True
        self._tick()

    def set_store_only(self, value: bool) -> None:
        """Auto screen: the engine fetches, the poller only reads the store."""
        self._store_only = value
        self.request_refresh()

    def on_worker_state_changed(self, event) -> None:
        if event.state is not WorkerState.ERROR:
            return
        if event.worker.group in {"refresh-normal", "refresh-store"}:
            if event.worker.group == "refresh-normal":
                self._normal_refreshing = False
                self._normal_started_at = None
            else:
                self._store_refreshing = False
            self._update_refresh_status()
            msg = str(event.worker.error)
            if msg != self._last_refresh_error:
                self._last_refresh_error = msg
                lane = "store refresh" if event.worker.group == "refresh-store" else "refresh"
                self.notify(
                    f"{lane.capitalize()} failed: {msg}", severity="warning", timeout=6
                )
        elif event.worker.group == "action":
            self.busy = False
            self._finish_action(False)
            self.notify(f"Action failed: {event.worker.error}", severity="error")
        elif event.worker.group == "engine":
            self.notify(
                f"Auto-switch engine stopped: {event.worker.error}",
                severity="error",
            )

    # -- mutating actions (single-flight, captured, off-thread) ---------------

    def _start_action(
        self,
        label: str,
        fn,
        *,
        show_output: bool = False,
        then: Callable[[bool], None] | None = None,
    ) -> bool:
        """Start one captured action off-thread; False if one is still running.

        Callers that mirror the action in the UI before it lands (the account
        lists move their cursor with a reorder) need to know it was refused,
        and ``then`` tells them how it ended — see :meth:`_finish_action`.
        """
        if self.busy:
            self.notify("Another action is still running", severity="warning")
            return False
        self.busy = True
        self._action_then = then
        self.run_worker(
            partial(self._action_blocking, label, fn, show_output),
            thread=True,
            group="action",
            exit_on_error=False,
            name=label,
        )
        return True

    def _action_blocking(self, label: str, fn, show_output: bool) -> None:
        result = run_action(fn)
        self.call_from_thread(self._action_done, label, result, show_output)

    def _finish_action(self, ok: bool) -> None:
        """Hand the outcome to the action's completion callback, exactly once.

        Actions are single-flight, so one slot holds it. Both endings run
        through here — the captured result in :meth:`_action_done`, and a
        worker crash in :meth:`on_worker_state_changed`, which ``run_action``
        never sees. A caller that undoes an optimistic UI change on failure
        would otherwise leave it standing on the path it cannot observe.
        """
        then, self._action_then = self._action_then, None
        if then is not None:
            then(ok)

    def _action_done(
        self, label: str, result: ActionResult, show_output: bool
    ) -> None:
        self.busy = False
        self.request_refresh()
        self._finish_action(result.ok)
        if not result.ok:
            self.push_screen(OutputModal(f"{label} — failed", result.output))
            return
        payload = result.payload or {}
        if "switched" in payload:
            if payload.get("switched"):
                to = payload.get("to") or {}
                target = to.get("email") or f"account {to.get('number')}"
                self.notify(f"Switched to {target}", title="Switch")
            else:
                reason = str(payload.get("reason") or "no switch performed")
                self.notify(reason, title="No switch", severity="warning")
            return
        if show_output and result.output.strip():
            self.push_screen(OutputModal(label, result.output))
        elif result.first_line:
            self.notify(result.first_line)

    # -- account operations ----------------------------------------------------

    def do_switch(self, number: str) -> None:
        self._start_action(
            f"Switch to account {number}",
            partial(self.switcher.switch_to, number, json_output=True),
        )

    def action_switch_best(self) -> None:
        self._start_action(
            "Switch (best)",
            partial(self.switcher.switch, strategy="best", json_output=True),
        )

    def do_toggle_disabled(self, number: str) -> None:
        """Hold the account out of auto-rotation, or return it — reads its
        current state from the live snapshot to pick the direction."""
        snap = self.snapshot
        acc = next(
            (a for a in (snap.accounts if snap else ()) if a.number == number), None
        )
        if acc is None:
            return
        target = not acc.disabled
        verb = "Disable" if target else "Enable"
        self._start_action(
            f"{verb} account {number}",
            partial(self.switcher.set_account_disabled, number, target),
        )

    def edit_identity(self, number: str) -> None:
        """Open the alias & name form for one account, prefilled.

        Every other account's alias goes in as taken, so the form can refuse
        a duplicate before it closes instead of failing the write after.
        """
        snap = self.snapshot
        accounts = snap.accounts if snap else ()
        acc = next((a for a in accounts if a.number == number), None)
        if acc is None:
            return
        taken = {a.alias: a.number for a in accounts if a.alias and a.number != number}
        self.push_screen(
            IdentityModal(
                number=number,
                email=acc.email,
                org_tag=acc.display_tag,
                alias=acc.alias,
                name=acc.display_name,
                taken_aliases=taken,
            ),
            partial(self._on_identity_form, number, acc.alias, acc.display_name),
        )

    def _on_identity_form(
        self, number: str, old_alias: str, old_name: str, form: IdentityForm | None
    ) -> None:
        if form is None:
            return
        if (form.alias, form.name) == (old_alias, old_name):
            self.notify("Nothing changed", timeout=2)
            return
        self._start_action(
            f"Alias & name for account {number}",
            partial(self._apply_identity, number, old_alias, old_name, form),
        )

    def _apply_identity(
        self, number: str, old_alias: str, old_name: str, form: IdentityForm
    ) -> None:
        """Write only what changed, then print the one line the notification
        shows. Runs inside ``run_action`` (off-thread, output captured)."""
        changes: list[str] = []
        if form.alias != old_alias:
            if form.alias:
                self.switcher.set_alias(number, form.alias)
                changes.append(f"alias '{form.alias}'")
            else:
                self.switcher.unset_alias(number)
                changes.append("alias removed")
        if form.name != old_name:
            if form.name:
                self.switcher.set_display_name(number, form.name)
                changes.append(f"name '{form.name}'")
            else:
                self.switcher.unset_display_name(number)
                changes.append("name removed")
        print(f"Account {number}: {', '.join(changes)}")

    def do_reorder(
        self, first: str, second: str, *, then: Callable[[bool], None] | None = None
    ) -> bool:
        """Exchange two accounts' slot numbers; the display order follows them.

        Every account list is in slot-number order, so "move one place up or
        down" is a swap with the neighbouring *slot* — which is what callers
        pass. Nothing is reported on success: the list visibly reorders on the
        refresh that follows. Returns False when the action was refused (so
        the caller can leave its cursor where it was), and calls ``then`` with
        the outcome once it lands.
        """
        return self._start_action(
            f"Reorder accounts {first} and {second}",
            partial(self.switcher.swap_accounts, first, second),
            then=then,
        )

    def confirm_remove(self, number: str, email: str) -> None:
        self.push_screen(
            ConfirmModal(
                f"Remove account {number} ({email})?\n\n"
                "Its stored credentials and config backup are deleted.",
                title="Remove account",
                yes_label="Remove",
            ),
            partial(self._on_remove_confirm, number),
        )

    def _on_remove_confirm(self, number: str, confirmed: bool | None) -> None:
        if confirmed:
            self._start_action(
                f"Remove account {number}",
                partial(self.switcher.remove_account, number, assume_yes=True),
            )

    def action_add_current(self) -> None:
        self.push_screen(
            ConfirmModal(
                "Back up the current Claude Code login as a managed account?\n\n"
                "If this account is already managed, its stored credentials "
                "are refreshed in place.",
                title="Add account",
                yes_label="Add",
            ),
            self._on_add_confirm,
        )

    def _on_add_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            self._start_action(
                "Add current login",
                partial(self.switcher.add_account),
                show_output=True,
            )

    def action_add_token(self) -> None:
        self.push_screen(AddTokenModal(), self._on_token_form)

    def _on_token_form(self, form: TokenForm | None) -> None:
        if form is None:
            return
        run = partial(
            self._start_action,
            "Add account from token",
            partial(
                self.switcher.add_account_from_token,
                token=form.token,
                email=form.email,
                slot=form.slot,
                assume_yes=True,
            ),
            show_output=True,
        )
        occupant = self._slot_occupant(form.slot)
        if occupant is not None:
            self.push_screen(
                ConfirmModal(
                    f"Slot {form.slot} is occupied by {occupant}. Overwrite?",
                    title="Overwrite slot",
                    yes_label="Overwrite",
                ),
                lambda confirmed: run() if confirmed else None,
            )
        else:
            run()

    def _slot_occupant(self, slot: int | None) -> str | None:
        if slot is None or self.snapshot is None:
            return None
        for acc in self.snapshot.accounts:
            if acc.number == str(slot):
                return acc.email
        return None

    # -- navigation -------------------------------------------------------------

    def action_refresh_full(self) -> None:
        self.request_refresh(full=True)
        self.notify("Refreshing usage…", timeout=2)

    def action_open_auto(self) -> None:
        if isinstance(self.screen, AutoScreen):
            return
        self.push_screen(AutoScreen())

    def action_open_watch(self) -> None:
        if isinstance(self.screen, WatchScreen):
            return
        self.push_screen(WatchScreen())

    # -- theme --------------------------------------------------------------

    def _resolved_theme(self) -> str:
        """Concrete 'dark'/'light' for the current setting; auto → detected → dark."""
        if self._theme_name == "auto":
            return self._detected or "dark"
        return self._theme_name

    def apply_theme(self, name: str) -> None:
        """Switch the live theme: TUI + captured printer output + persistence.

        ``name`` is the setting; ``auto`` resolves against the pre-driver
        detection (never re-probes mid-session)."""
        self._theme_name = name
        resolved = self._resolved_theme()
        self.theme = f"cswap-{resolved}"
        printer.set_theme(resolved)
        try:
            set_setting(self.switcher.backup_dir, "ui.theme", name)
        except Exception as exc:  # persistence is best-effort; never crash the UI
            self.notify(f"Could not save theme: {exc}", severity="warning")

    def action_toggle_theme(self) -> None:
        order = ("dark", "light", "auto")
        nxt = order[(order.index(self._theme_name) + 1) % len(order)]
        self.apply_theme(nxt)
        self.notify(f"Theme: {nxt}")
