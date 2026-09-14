#!/usr/bin/python3
"""GTK launcher for the NavRL 3-D simulator."""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import time

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk  # noqa: E402


def _load_launcher_common():
    path = Path(__file__).with_name("navrl_3d_launcher_common.py")
    spec = importlib.util.spec_from_file_location("navrl_3d_launcher_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_common = _load_launcher_common()
APP = _common.APP
CHECKPOINT_KIND_SHORT = _common.CHECKPOINT_KIND_SHORT
DEFAULT_DENSITY_MAX = _common.DEFAULT_DENSITY_MAX
DEFAULT_DENSITY_MIN = _common.DEFAULT_DENSITY_MIN
DEFAULT_DRONE_SPEED = _common.DEFAULT_DRONE_SPEED
DEFAULT_NUM_TRIALS = _common.DEFAULT_NUM_TRIALS
DEFAULT_TARGET_SPEED = _common.DEFAULT_TARGET_SPEED
RL_RUNS = _common.RL_RUNS
VIEWER_CONTROL_CHIPS = _common.VIEWER_CONTROL_CHIPS
find_recent_checkpoints = _common.find_recent_checkpoints
inspect_checkpoint = _common.inspect_checkpoint
validate_checkpoint_info = _common.validate_checkpoint_info


CSS = b"""
* {
  font-family: "Cantarell", "Segoe UI", "Noto Sans", "Helvetica Neue", sans-serif;
}
window, .app-root {
  background-color: #e9edf0;
}
.header {
  background-image: linear-gradient(135deg, #071018 0%, #0f2430 48%, #0d4f58 100%);
  padding: 34px 36px 28px 36px;
}
.kicker {
  color: #7fe0e7;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  font-weight: 600;
}
.title {
  color: #f2f8fb;
  font-size: 34px;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.subtitle {
  color: #a8bac8;
  font-size: 15px;
  margin-top: 2px;
}
.chip-row {
  margin-top: 14px;
}
.chip {
  background-color: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(127, 224, 231, 0.28);
  border-radius: 999px;
  padding: 5px 12px;
  margin-right: 8px;
}
.chip-label {
  color: #d7e7ef;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 11px;
  font-weight: 600;
}
.content {
  padding: 24px 28px 18px 28px;
}
.card {
  background-color: #f4f6f8;
  border: 1px solid #cfd6dc;
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: 0 10px 28px rgba(17, 24, 32, 0.05);
}
.card-title {
  color: #111820;
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 2px;
}
.card-sub {
  color: #586576;
  font-size: 13px;
  margin-bottom: 14px;
}
.field-label {
  color: #586576;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 10.5px;
  letter-spacing: 0.06em;
  font-weight: 600;
  margin-top: 2px;
}
.field-value {
  color: #111820;
  font-size: 14px;
  font-weight: 600;
}
.hint {
  color: #7d8896;
  font-size: 12px;
}
.status-box {
  background-color: #ffffff;
  border: 1px solid #dde3e8;
  border-left: 4px solid #8b97a4;
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 8px;
  margin-bottom: 10px;
}
.status-text {
  color: #586576;
  font-size: 13px;
}
.status-ok {
  border-left-color: #2f9e60;
}
.status-ok .status-text {
  color: #1f7a49;
  font-weight: 600;
}
.status-warn {
  border-left-color: #c99a2e;
}
.status-warn .status-text {
  color: #9a7418;
  font-weight: 600;
}
.status-error {
  border-left-color: #d16a37;
}
.status-error .status-text {
  color: #b04f22;
  font-weight: 600;
}
.recent-list {
  background-color: #ffffff;
  border: 1px solid #dde3e8;
  border-radius: 10px;
}
.recent-list row {
  padding: 0;
  border-bottom: 1px solid #eef2f5;
}
.recent-list row:last-child {
  border-bottom: none;
}
.recent-list row:selected {
  background-color: rgba(13, 151, 164, 0.12);
}
.recent-name {
  color: #111820;
  font-size: 13px;
  font-weight: 600;
}
.recent-path {
  color: #7d8896;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 10.5px;
}
.controls-card {
  margin-top: 16px;
}
.chip-key {
  background-color: #111820;
  color: #f2f8fb;
  border-radius: 6px;
  padding: 4px 8px;
  margin-right: 6px;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 11px;
  font-weight: 700;
}
.chip-desc {
  color: #586576;
  font-size: 12px;
  margin-right: 16px;
}
.footer {
  padding: 8px 28px 24px 28px;
}
.footer-note {
  color: #7d8896;
  font-size: 12px;
}
button {
  border-radius: 10px;
  padding: 11px 18px;
  font-size: 14px;
  font-weight: 600;
}
.secondary {
  background-color: #ffffff;
  color: #243041;
  border: 1px solid #cfd6dc;
}
.secondary:hover {
  background-color: #eef2f5;
}
.primary {
  background-image: linear-gradient(135deg, #0d97a4 0%, #0a6f79 100%);
  color: #ffffff;
  border: none;
  padding-left: 22px;
  padding-right: 22px;
}
.primary:hover {
  background-image: linear-gradient(135deg, #0fb3c2 0%, #0d8591 100%);
}
spinbutton, entry, filechooserbutton {
  background-color: #ffffff;
  border: 1px solid #cfd6dc;
  border-radius: 8px;
  padding: 4px 8px;
}
.filepicker {
  margin-top: 4px;
}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-python", required=True)
    return parser.parse_args()


def _label(text, css_class, xalign=0.0, markup=False):
    widget = Gtk.Label()
    if markup:
        widget.set_markup(text)
    else:
        widget.set_label(text)
    widget.set_xalign(xalign)
    widget.set_line_wrap(True)
    widget.get_style_context().add_class(css_class)
    return widget


def _mono(text, css_class="field-value"):
    label = _label(text, css_class)
    label.get_style_context().add_class("mono")
    return label


class Launcher:
    def __init__(self, runtime_python):
        self.runtime_python = str(Path(runtime_python).expanduser().resolve())
        self.mode = None
        self.recent_paths = find_recent_checkpoints(limit=8)
        self.window = Gtk.Window(title="NavRL 3D Simulator")
        self.window.set_default_size(1120, 820)
        self.window.set_size_request(980, 740)
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.connect("destroy", Gtk.main_quit)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        root.get_style_context().add_class("app-root")
        self.window.add(root)

        root.pack_start(self._header(), False, False, 0)
        root.pack_start(self._body(), True, True, 0)
        root.pack_start(self._footer(), False, False, 0)

    def _header(self):
        header = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        header.get_style_context().add_class("header")
        header.pack_start(_label("MOTAR · NAVRL 3D", "kicker"), False, False, 0)
        header.pack_start(_label("Perception Evaluator", "title"), False, False, 0)
        header.pack_start(
            _label(
                "Real Isaac Gym scene · RGB-D + LiDAR perception · Transformer policy playback",
                "subtitle",
            ),
            False,
            False,
            0,
        )
        chips = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        chips.get_style_context().add_class("chip-row")
        for text in ("574D Transformer", "Generalized trials", "Live HUD"):
            chip = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            chip.get_style_context().add_class("chip")
            chip.pack_start(_label(text, "chip-label"), False, False, 0)
            chips.pack_start(chip, False, False, 0)
        header.pack_start(chips, False, False, 0)
        return header

    def _body(self):
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content.get_style_context().add_class("content")

        columns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)
        columns.pack_start(self._evaluation_card(), True, True, 0)
        columns.pack_start(self._policy_card(), True, True, 0)
        content.pack_start(columns, True, True, 0)
        content.pack_start(self._controls_card(), False, False, 0)
        return content

    def _card_shell(self, title, subtitle):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        outer.get_style_context().add_class("card")
        outer.pack_start(_label(title, "card-title"), False, False, 0)
        outer.pack_start(_label(subtitle, "card-sub"), False, False, 0)
        return outer

    def _form_row(self, caption, widget, hint=None):
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        row.pack_start(_label(caption.upper(), "field-label"), False, False, 0)
        row.pack_start(widget, False, False, 0)
        if hint:
            row.pack_start(_label(hint, "hint"), False, False, 0)
        return row

    def _evaluation_card(self):
        box = self._card_shell(
            "Evaluation setup",
            "Randomize obstacle density, drone spawn, and target placement each trial.",
        )

        stats = Gtk.Grid(column_spacing=16, row_spacing=14)
        stats.set_column_homogeneous(True)

        self.num_trials = Gtk.SpinButton.new_with_range(1, 100, 1)
        self.num_trials.set_value(DEFAULT_NUM_TRIALS)
        stats.attach(self._form_row("Trials", self.num_trials), 0, 0, 1, 1)

        density_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.density_min = Gtk.SpinButton.new_with_range(0, 200, 5)
        self.density_min.set_value(DEFAULT_DENSITY_MIN)
        self.density_max = Gtk.SpinButton.new_with_range(0, 200, 5)
        self.density_max.set_value(DEFAULT_DENSITY_MAX)
        density_row.pack_start(self.density_min, False, False, 0)
        density_row.pack_start(_label("to", "hint"), False, False, 0)
        density_row.pack_start(self.density_max, False, False, 0)
        stats.attach(
            self._form_row("Density range (bars)", density_row, "Sampled uniformly each trial"),
            1,
            0,
            1,
            1,
        )

        self.target_speed = Gtk.SpinButton.new_with_range(0.0, 2.5, 0.25)
        self.target_speed.set_value(DEFAULT_TARGET_SPEED)
        self.target_speed.set_digits(2)
        stats.attach(
            self._form_row("Target speed (m/s)", self.target_speed, "Must be > 0 for policy mode"),
            0,
            1,
            1,
            1,
        )

        self.drone_speed = Gtk.SpinButton.new_with_range(0.25, 3.0, 0.25)
        self.drone_speed.set_value(DEFAULT_DRONE_SPEED)
        self.drone_speed.set_digits(2)
        stats.attach(
            self._form_row("Drone max speed (m/s)", self.drone_speed),
            1,
            1,
            1,
            1,
        )
        box.pack_start(stats, False, False, 0)
        return box

    def _policy_card(self):
        box = self._card_shell(
            "Policy model",
            "Pick a trained checkpoint. Compatibility is checked before launch.",
        )

        picker = Gtk.FileChooserButton.new(
            "Choose checkpoint (.pth)", Gtk.FileChooserAction.OPEN
        )
        picker.get_style_context().add_class("filepicker")
        if RL_RUNS.is_dir():
            picker.set_current_folder(str(RL_RUNS))
        file_filter = Gtk.FileFilter()
        file_filter.set_name("PyTorch checkpoint (*.pth)")
        file_filter.add_pattern("*.pth")
        picker.add_filter(file_filter)
        picker.connect("file-set", self._checkpoint_changed)
        self.checkpoint = picker
        box.pack_start(self._form_row("Checkpoint file", picker), False, False, 0)

        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        status_box.get_style_context().add_class("status-box")
        self.status = _label("Select a checkpoint to inspect model compatibility.", "status-text")
        status_box.pack_start(self.status, False, False, 0)
        self.status_box = status_box
        box.pack_start(status_box, False, False, 0)

        box.pack_start(_label("RECENT CHECKPOINTS", "field-label"), False, False, 6)
        self.recent_list = Gtk.ListBox()
        self.recent_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.recent_list.get_style_context().add_class("recent-list")
        if self.recent_paths:
            for path in self.recent_paths:
                self.recent_list.add(self._recent_row(path))
        else:
            empty = Gtk.ListBoxRow()
            empty.set_sensitive(False)
            empty.add(_label("No recent checkpoints found under runs/ or checkpoints_saved/", "hint"))
            self.recent_list.add(empty)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(150)
        scroll.add(self.recent_list)
        box.pack_start(scroll, True, True, 0)
        self.recent_list.connect("row-activated", self._recent_activated)
        return box

    def _recent_row(self, path: Path):
        row = Gtk.ListBoxRow()
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        inner.set_border_width(10)
        inner.pack_start(_label(path.name, "recent-name"), False, False, 0)
        parent = str(path.parent)
        if len(parent) > 72:
            parent = "…" + parent[-69:]
        inner.pack_start(_label(parent, "recent-path"), False, False, 0)
        row.add(inner)
        row.path_value = str(path)
        return row

    def _controls_card(self):
        box = self._card_shell(
            "In-viewer controls",
            "Adjust speeds and switch manual control while the Isaac Gym window is open.",
        )
        box.get_style_context().add_class("controls-card")
        flow = Gtk.FlowBox()
        flow.set_selection_mode(Gtk.SelectionMode.NONE)
        flow.set_max_children_per_line(4)
        flow.set_column_spacing(18)
        flow.set_row_spacing(10)
        for key, desc in VIEWER_CONTROL_CHIPS:
            item = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
            key_label = _label(key, "chip-key")
            item.pack_start(key_label, False, False, 0)
            item.pack_start(_label(desc, "chip-desc"), False, False, 0)
            flow.add(item)
        box.pack_start(flow, False, False, 0)
        return box

    def _footer(self):
        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        footer.get_style_context().add_class("footer")
        footer.pack_start(
            _label("Results are saved to results/general_eval_results.json after the run.", "footer-note"),
            True,
            True,
            0,
        )
        manual = Gtk.Button(label="Manual preview")
        manual.get_style_context().add_class("secondary")
        manual.connect("clicked", self._finish, "manual")
        policy = Gtk.Button(label="Start evaluation")
        policy.get_style_context().add_class("primary")
        policy.connect("clicked", self._finish, "policy")
        footer.pack_end(policy, False, False, 0)
        footer.pack_end(manual, False, False, 0)
        return footer

    def _set_status(self, text, css_class):
        context = self.status_box.get_style_context()
        for old in ("status-ok", "status-warn", "status-error"):
            context.remove_class(old)
        if css_class in ("status-ok", "status-warn", "status-error"):
            context.add_class(css_class)
        self.status.set_text(text)

    def _checkpoint_changed(self, _widget):
        filename = self.checkpoint.get_filename()
        if not filename:
            self._set_status("No checkpoint selected.", "hint")
            return None
        info = inspect_checkpoint(filename, python=self.runtime_python)
        label_text, error = validate_checkpoint_info(info)
        if error:
            self._set_status(error, "status-error")
            return None
        css = "status-ok" if info["kind"] == "transformer" else "status-warn"
        self._set_status(label_text, css)
        return info

    def _recent_activated(self, _listbox, row):
        value = getattr(row, "path_value", None)
        if not value:
            return
        self.checkpoint.set_filename(value)
        self._checkpoint_changed(self.checkpoint)

    def _error(self, title, message):
        dialog = Gtk.MessageDialog(
            transient_for=self.window,
            flags=Gtk.DialogFlags.MODAL,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text=title,
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def _finish(self, _button, mode):
        if int(self.density_min.get_value()) > int(self.density_max.get_value()):
            self._error("Invalid density range", "Minimum bars must be <= maximum bars.")
            return
        if mode == "policy":
            filename = self.checkpoint.get_filename()
            if not filename:
                self._error("Checkpoint required", "Select a trained .pth checkpoint.")
                return
            if self.target_speed.get_value() <= 0.0:
                self._error("Invalid target speed", "Policy evaluation requires a moving target.")
                return
            if self._checkpoint_changed(self.checkpoint) is None:
                self._error("Incompatible checkpoint", self.status.get_text())
                return
        self.mode = mode
        self.window.destroy()

    def run(self):
        self.window.show_all()
        Gtk.main()
        if self.mode is None:
            return 0
        command = [
            self.runtime_python,
            str(APP),
            "--target-speed",
            str(self.target_speed.get_value()),
            "--drone-speed",
            str(self.drone_speed.get_value()),
            "--num-trials",
            str(int(self.num_trials.get_value())),
            "--density-min",
            str(int(self.density_min.get_value())),
            "--density-max",
            str(int(self.density_max.get_value())),
        ]
        if self.mode == "manual":
            command.append("--manual")
        else:
            command.extend(("--checkpoint", self.checkpoint.get_filename()))
        repo = APP.parents[2]
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        env["PYTHONPATH"] = str(repo) + (
            (":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
        )
        return subprocess.call(command, cwd=str(repo), env=env)


def main():
    args = parse_args()
    provider = Gtk.CssProvider()
    provider.load_from_data(CSS)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    return Launcher(args.runtime_python).run()


if __name__ == "__main__":
    raise SystemExit(main())
