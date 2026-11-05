"""Modal screens: confirmations, token entry, alias & name, captured output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from claude_swap.models import (
    DISPLAY_NAME_MAX_LEN,
    normalize_alias,
    normalize_display_name,
)


class ConfirmModal(ModalScreen[bool]):
    """Yes/No confirmation. Dismisses with True only on explicit confirm.

    Keyboard-first: ←/→ move between the buttons (Enter presses the focused
    one), y/n answer directly, Esc cancels. Clicking still works.
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes", show=False),
        Binding("n,escape", "cancel", "No", show=False),
        Binding("left", "app.focus_previous", show=False),
        Binding("right", "app.focus_next", show=False),
    ]

    def __init__(
        self, message: str, *, title: str = "Confirm", yes_label: str = "Yes"
    ) -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._yes_label = yes_label

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Label(self._title, classes="modal-title")
            yield Static(self._message, classes="modal-body")
            with Horizontal(classes="modal-buttons"):
                yield Button(self._yes_label, id="yes")
                yield Button("Cancel", id="no")
            yield Static(
                f"← → · enter  ·  y {self._yes_label.lower()}  ·  n / esc cancel",
                classes="modal-hint",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


@dataclass
class TokenForm:
    """What the add-token modal collects."""

    token: str
    email: str | None
    slot: int | None


class AddTokenModal(ModalScreen["TokenForm | None"]):
    """Collects a setup-token/API key, optional email label, optional slot.

    ←/→ only reach the screen when a Button is focused (a focused Input
    consumes them for cursor movement), so they safely double as button
    navigation.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left", "app.focus_previous", show=False),
        Binding("right", "app.focus_next", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Label("Add account from token", classes="modal-title")
            yield Static(
                "OAuth setup-token (sk-ant-oat…) or Console API key "
                "(sk-ant-api…, sk-ant-usr…); the type is auto-detected.",
                classes="modal-body",
            )
            yield Input(password=True, placeholder="token (required)", id="token")
            yield Input(placeholder="email label (optional)", id="email")
            yield Input(placeholder="slot number (optional)", id="slot", type="integer")
            yield Static("", id="form-error", classes="form-error")
            with Horizontal(classes="modal-buttons"):
                yield Button("Add", id="add")
                yield Button("Cancel", id="cancel")
            yield Static(
                "enter add  ·  tab next field  ·  esc cancel",
                classes="modal-hint",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        token = self.query_one("#token", Input).value.strip()
        email = self.query_one("#email", Input).value.strip() or None
        slot_raw = self.query_one("#slot", Input).value.strip()
        if not token:
            self.query_one("#form-error", Static).update("Token is required.")
            return
        slot: int | None = None
        if slot_raw:
            try:
                slot = int(slot_raw)
            except ValueError:
                self.query_one("#form-error", Static).update(
                    "Slot must be a number."
                )
                return
            if slot < 1:
                self.query_one("#form-error", Static).update("Slot must be >= 1.")
                return
        self.dismiss(TokenForm(token=token, email=email, slot=slot))

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True)
class IdentityForm:
    """What the alias-and-name modal collects, already normalized.

    ``""`` means "none": the caller removes whatever the account had.
    """

    alias: str
    name: str


class IdentityModal(ModalScreen["IdentityForm | None"]):
    """Edit one account's alias and display name in a single form.

    Prefilled with the current values; an emptied field removes that value.
    Every rule is checked here, before the form closes, so a typo costs a
    correction rather than a failed action and a retype: the alias format and
    uniqueness (``taken_aliases`` maps every *other* account's alias to its
    number), and the display-name rules. The store re-checks both on write,
    so a race with another process still fails safely, just later.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left", "app.focus_previous", show=False),
        Binding("right", "app.focus_next", show=False),
    ]

    def __init__(
        self,
        *,
        number: str,
        email: str,
        org_tag: str,
        alias: str,
        name: str,
        taken_aliases: Mapping[str, str],
    ) -> None:
        super().__init__()
        self._number = number
        self._email = email
        self._org_tag = org_tag
        self._alias = alias
        self._name = name
        self._taken = dict(taken_aliases)

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Label(
                f"Alias & name · account {self._number}", classes="modal-title"
            )
            yield Static(f"{self._email}\n[{self._org_tag}]", classes="modal-body")
            yield Label("alias", classes="field-label")
            yield Input(
                value=self._alias,
                placeholder="e.g. luxoft-main",
                id="alias",
            )
            yield Static(
                "lowercase · letters, digits, - _ . · usable in cswap switch",
                classes="field-hint",
            )
            yield Label("name", classes="field-label")
            yield Input(
                value=self._name,
                placeholder="e.g. Luxoft – main",
                id="name",
                max_length=DISPLAY_NAME_MAX_LEN,
            )
            yield Static(
                f"anything · up to {DISPLAY_NAME_MAX_LEN} characters · display only",
                classes="field-hint",
            )
            yield Static("", id="form-error", classes="form-error")
            with Horizontal(classes="modal-buttons"):
                yield Button("Save", id="save")
                yield Button("Cancel", id="cancel")
            yield Static(
                "empty field = remove  ·  enter save  ·  tab next  ·  esc cancel",
                classes="modal-hint",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _error(self, message: str, field: str) -> None:
        self.query_one("#form-error", Static).update(message)
        self.query_one(f"#{field}", Input).focus()

    def _submit(self) -> None:
        alias_raw = self.query_one("#alias", Input).value.strip()
        name_raw = self.query_one("#name", Input).value.strip()
        alias = ""
        if alias_raw:
            try:
                alias = normalize_alias(alias_raw)
            except ValueError as e:
                self._error(f"Alias: {e}", "alias")
                return
            owner = self._taken.get(alias)
            if owner is not None:
                self._error(
                    f"Alias: '{alias}' is already used by account {owner}", "alias"
                )
                return
        name = ""
        if name_raw:
            try:
                name = normalize_display_name(name_raw)
            except ValueError as e:
                self._error(f"Name: {e}", "name")
                return
        self.dismiss(IdentityForm(alias=alias, name=name))

    def action_cancel(self) -> None:
        self.dismiss(None)


class OutputModal(ModalScreen[None]):
    """Scrollable display of captured (ANSI-colored) action output."""

    BINDINGS = [Binding("escape,q,enter", "dismiss_modal", "Close", show=False)]

    def __init__(self, title: str, output: str) -> None:
        super().__init__()
        self._title = title
        self._output = output

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box modal-box-wide"):
            yield Label(self._title, classes="modal-title")
            with VerticalScroll(classes="modal-output"):
                yield Static(Text.from_ansi(self._output.rstrip() or "(no output)"))
            with Horizontal(classes="modal-buttons"):
                yield Button("Close", id="close")
            yield Static("esc close", classes="modal-hint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
