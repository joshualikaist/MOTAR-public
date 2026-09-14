#!/usr/bin/env python3
"""Native interactive NavRL 3-D application.

This is not a Three.js mock-up: it runs the real Isaac Gym task, Warp LiDAR, RGB-D perception,
and (when selected) an rl_games Transformer checkpoint. With no checkpoint it opens a manual
sensor/environment demo so the application can be inspected before policy training finishes.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys

from aerial_gym.apps.navrl_3d_launcher_common import (
    DEFAULT_DENSITY_MAX,
    DEFAULT_DENSITY_MIN,
    DEFAULT_DRONE_SPEED,
    DEFAULT_NUM_TRIALS,
    DEFAULT_TARGET_SPEED,
    RL_DIR,
    VIEWER_CONTROL_CHIPS,
    apply_runtime_environment,
    configure_policy_checkpoint,
    find_recent_checkpoints,
    inspect_checkpoint,
    validate_checkpoint_info,
)


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def _args():
    parser = argparse.ArgumentParser(description="NavRL interactive Isaac Gym application")
    parser.add_argument("--checkpoint", type=Path, help="supported NavRL .pth checkpoint")
    parser.add_argument("--manual", action="store_true", help="run without a policy checkpoint")
    parser.add_argument("--bars", type=int, default=48, help=argparse.SUPPRESS)
    parser.add_argument("--target-speed", type=float, default=DEFAULT_TARGET_SPEED)
    parser.add_argument("--drone-speed", type=float, default=DEFAULT_DRONE_SPEED)
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--density-min", type=int, default=DEFAULT_DENSITY_MIN)
    parser.add_argument("--density-max", type=int, default=DEFAULT_DENSITY_MAX)
    parser.add_argument("--results-json", type=Path, help="path for generalized eval summary JSON")
    parser.add_argument("--num-envs", type=int, default=1)
    return parser.parse_args()


def _setup_dialog(args):
    """Large high-DPI launcher; live controls continue inside the 3-D window."""
    if args.manual or args.checkpoint is not None:
        return args
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise SystemExit("tkinter is unavailable; pass --manual or --checkpoint PATH") from exc

    root = tk.Tk()
    root.title("NavRL 3D Simulator")
    root.geometry("1120x820")
    root.minsize(980, 740)
    root.configure(bg="#E9EDF0")
    root.tk.call("tk", "scaling", 1.45)

    ACCENT = "#0D97A4"
    INK = "#111820"
    SLATE = "#586576"
    PANEL = "#F4F6F8"
    LINE = "#CFD6DC"

    from tkinter import font as tkfont

    base_font = tkfont.nametofont("TkDefaultFont")

    def sized_font(size, bold=False):
        value = base_font.copy()
        value.configure(size=size, weight="bold" if bold else "normal")
        return value

    title_font = sized_font(27, bold=True)
    section_font = sized_font(15, bold=True)
    card_title_font = sized_font(14, bold=True)
    body_font = sized_font(12)
    copy_font = sized_font(11)
    hint_font = sized_font(10)
    hint_bold_font = sized_font(10, bold=True)
    button_font = sized_font(11, bold=True)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Root.TFrame", background="#E9EDF0")
    style.configure("Card.TLabelframe", background=PANEL, borderwidth=1, relief="solid")
    style.configure(
        "Card.TLabelframe.Label",
        background=PANEL,
        foreground=INK,
        font=card_title_font,
    )
    style.configure("TLabel", background=PANEL, foreground=INK, font=body_font)
    style.configure("Hint.TLabel", foreground="#7D8896", font=hint_font)
    style.configure("TEntry", font=copy_font, padding=7)
    style.configure("TSpinbox", font=copy_font, padding=5)
    style.configure("TButton", font=button_font, padding=(14, 10))
    style.configure(
        "Accent.TButton", background=ACCENT, foreground="#FFFFFF", borderwidth=0
    )
    style.map("Accent.TButton", background=[("active", "#0A6F79")])
    style.configure(
        "Manual.TButton", background="#FFFFFF", foreground="#243041", borderwidth=1
    )

    checkpoint = tk.StringVar(value="")
    target_speed = tk.StringVar(value=str(args.target_speed))
    drone_speed = tk.StringVar(value=str(args.drone_speed))
    num_trials = tk.StringVar(value=str(args.num_trials))
    density_min = tk.StringVar(value=str(args.density_min))
    density_max = tk.StringVar(value=str(args.density_max))
    compatibility = tk.StringVar(value="Select a checkpoint to inspect model compatibility.")
    compatibility_color = {"value": "#66758C"}
    result = {"mode": None}
    recent_paths = find_recent_checkpoints(limit=8)
    recent_index_map = {}

    def inspect_selected(show_error=False):
        raw = checkpoint.get().strip()
        if not raw:
            set_compatibility("No checkpoint selected. Manual preview is available.", "#66758C")
            return None
        path = Path(raw).expanduser()
        if not path.is_file():
            set_compatibility("File not found.", "#DC2626")
            return None
        info = inspect_checkpoint(path)
        label, error = validate_checkpoint_info(info)
        if error:
            set_compatibility(error, "#DC2626")
            if show_error:
                messagebox.showerror("Incompatible checkpoint", error)
            return None
        color = "#15803D" if info["kind"] == "transformer" else "#B45309"
        set_compatibility(label, color)
        return info

    def browse():
        value = filedialog.askopenfilename(
            title="Perception Transformer checkpoint",
            initialdir=str(RL_DIR / "runs"),
            filetypes=(("PyTorch checkpoint", "*.pth"), ("All files", "*")),
        )
        if value:
            checkpoint.set(value)
            inspect_selected()

    def choose_recent(_event=None):
        selection = recent_list.curselection()
        if not selection:
            return
        path = recent_index_map.get(selection[0])
        if path is None:
            return
        checkpoint.set(str(path))
        inspect_selected()

    def finish(mode):
        try:
            args.target_speed = float(target_speed.get())
            args.drone_speed = float(drone_speed.get())
            args.num_trials = max(1, int(num_trials.get()))
            args.density_min = int(density_min.get())
            args.density_max = int(density_max.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Speeds, trials, and density must be numbers.")
            return
        if args.density_min > args.density_max:
            messagebox.showerror("Invalid density range", "Minimum bars must be <= maximum bars.")
            return
        if mode == "policy" and args.target_speed <= 0.0:
            messagebox.showerror("Invalid input", "Policy evaluation requires target speed above zero.")
            return
        args.num_envs = 1
        if mode == "policy":
            path = Path(checkpoint.get()).expanduser()
            if not path.is_file():
                messagebox.showerror("Checkpoint required", "Select a trained .pth checkpoint.")
                return
            args.checkpoint = path
            try:
                configure_policy_checkpoint(args)
            except ValueError as exc:
                messagebox.showerror("Checkpoint compatibility error", str(exc))
                inspect_selected(show_error=False)
                return
        else:
            args.manual = True
        result["mode"] = mode
        root.destroy()

    header = tk.Frame(root, bg="#071018", height=148)
    header.pack(fill="x")
    header.pack_propagate(False)
    accent = tk.Frame(header, bg=ACCENT, width=5)
    accent.pack(side="left", fill="y")
    head_body = tk.Frame(header, bg="#071018")
    head_body.pack(side="left", fill="both", expand=True)
    tk.Label(
        head_body, text="MOTAR · NAVRL 3D", bg="#071018", fg="#7FE0E7",
        font=sized_font(10, bold=True),
    ).pack(anchor="w", padx=28, pady=(22, 0))
    tk.Label(
        head_body, text="Perception Evaluator", bg="#071018", fg="#F2F8FB",
        font=title_font,
    ).pack(anchor="w", padx=28, pady=(2, 0))
    tk.Label(
        head_body,
        text="Real Isaac Gym scene · RGB-D + LiDAR perception · Transformer policy playback",
        bg="#071018", fg="#A8BAC8", font=copy_font,
    ).pack(anchor="w", padx=28, pady=(4, 10))
    chips = tk.Frame(head_body, bg="#071018")
    chips.pack(anchor="w", padx=28, pady=(0, 16))
    for text in ("574D Transformer", "Generalized trials", "Live HUD"):
        tk.Label(
            chips, text="  %s  " % text, bg="#122430", fg="#D7E7EF",
            font=hint_bold_font, highlightbackground="#3A6670", highlightthickness=1,
        ).pack(side="left", padx=(0, 8))

    main = ttk.Frame(root, padding=(28, 22, 28, 10), style="Root.TFrame")
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)

    scene = ttk.LabelFrame(main, text="  Evaluation setup  ", style="Card.TLabelframe", padding=18)
    scene.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
    model = ttk.LabelFrame(main, text="  Policy model  ", style="Card.TLabelframe", padding=18)
    model.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

    tk.Label(
        scene,
        text="Generalized evaluation",
        bg=PANEL, fg=INK, font=section_font,
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    tk.Label(
        scene,
        text="Randomize obstacle density, drone spawn, and target placement each trial.",
        bg=PANEL, fg=SLATE, justify="left", font=copy_font,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 18))

    ttk.Label(scene, text="Trials").grid(row=2, column=0, sticky="w")
    ttk.Spinbox(scene, from_=1, to=100, increment=1, textvariable=num_trials, width=8).grid(
        row=2, column=1, sticky="e"
    )
    ttk.Label(scene, text="Density min / max bars").grid(row=3, column=0, sticky="w", pady=(10, 0))
    density_row = ttk.Frame(scene, style="TFrame")
    density_row.grid(row=3, column=1, sticky="e", pady=(10, 0))
    ttk.Spinbox(
        density_row, from_=0, to=200, increment=5, textvariable=density_min, width=6
    ).pack(side="left")
    ttk.Label(density_row, text=" to ", style="Hint.TLabel").pack(side="left")
    ttk.Spinbox(
        density_row, from_=0, to=200, increment=5, textvariable=density_max, width=6
    ).pack(side="left")

    ttk.Label(scene, text="Target speed").grid(row=4, column=0, sticky="w", pady=(14, 0))
    ttk.Spinbox(
        scene, from_=0.0, to=2.5, increment=0.25, textvariable=target_speed, width=8
    ).grid(row=4, column=1, sticky="e", pady=(14, 0))
    ttk.Label(scene, text="m/s | target always moves in policy mode", style="Hint.TLabel").grid(
        row=5, column=0, columnspan=2, sticky="w", pady=(4, 12)
    )
    ttk.Label(scene, text="Drone max speed").grid(row=6, column=0, sticky="w")
    ttk.Spinbox(
        scene, from_=0.25, to=3.0, increment=0.25, textvariable=drone_speed, width=8
    ).grid(row=6, column=1, sticky="e")

    ttk.Label(model, text="Trained checkpoint").pack(anchor="w")
    ck_row = ttk.Frame(model, style="TFrame")
    ck_row.pack(fill="x", pady=(7, 10))
    ttk.Entry(ck_row, textvariable=checkpoint).pack(side="left", fill="x", expand=True)
    ttk.Button(ck_row, text="Browse", command=browse).pack(side="left", padx=(8, 0))
    status_box = tk.Frame(model, bg="#FFFFFF", highlightbackground=LINE, highlightthickness=1)
    status_box.pack(fill="x", pady=(2, 12))
    status_accent = tk.Frame(status_box, bg="#8B97A4", width=4)
    status_accent.pack(side="left", fill="y")
    status_label = tk.Label(
        status_box, textvariable=compatibility, bg="#FFFFFF", fg=SLATE,
        justify="left", anchor="w", wraplength=390, font=hint_bold_font,
    )
    status_label.pack(side="left", fill="x", expand=True, padx=12, pady=11)

    def set_compatibility(text, color):
        compatibility.set(text)
        compatibility_color["value"] = color
        status_label.configure(fg=color)
        status_accent.configure(bg=color if color.startswith("#") else ACCENT)

    ttk.Label(model, text="Recent checkpoints", style="Hint.TLabel").pack(anchor="w")
    recent_list = tk.Listbox(model, height=5, font=copy_font, activestyle="dotbox")
    recent_list.pack(fill="x", pady=(4, 10))
    recent_list.bind("<<ListboxSelect>>", choose_recent)
    if recent_paths:
        for path in recent_paths:
            line_idx = recent_list.size()
            recent_list.insert("end", path.name)
            recent_index_map[line_idx] = path
            recent_list.insert("end", "  %s" % path.parent)
            recent_list.insert("end", "")
    else:
        recent_list.insert("end", "(no recent checkpoints found)")

    controls = ttk.LabelFrame(main, text="  In-viewer controls  ", style="Card.TLabelframe", padding=14)
    controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(18, 0))
    chip_row = tk.Frame(controls, bg=PANEL)
    chip_row.pack(fill="x", pady=2)
    for key, desc in VIEWER_CONTROL_CHIPS:
        key_lbl = tk.Label(
            chip_row, text=" %s " % key, bg=INK, fg="#F2F8FB", font=hint_bold_font,
        )
        key_lbl.pack(side="left", padx=(0, 4))
        tk.Label(chip_row, text=desc, bg=PANEL, fg=SLATE, font=hint_font).pack(
            side="left", padx=(0, 14)
        )

    footer = ttk.Frame(root, padding=(28, 4, 28, 22), style="Root.TFrame")
    footer.pack(fill="x")
    tk.Label(
        footer,
        text="Results are saved to results/general_eval_results.json after the run.",
        background="#E9EDF0",
        foreground="#7D8896",
        font=hint_font,
    ).pack(side="left")
    ttk.Button(
        footer, text="Manual preview", style="Manual.TButton",
        command=lambda: finish("manual"),
    ).pack(side="right", padx=(8, 0))
    ttk.Button(
        footer, text="Start evaluation", style="Accent.TButton",
        command=lambda: finish("policy"),
    ).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    if result["mode"] is None:
        raise SystemExit(0)
    return args


def _run_manual(args):
    import isaacgym  # noqa: F401
    import torch
    from aerial_gym.registry.task_registry import task_registry

    apply_runtime_environment(args)
    task = task_registry.make_task("navrl_task", headless=False, num_envs=1, use_warp=True)
    task.reset()
    task._interactive_manual = True
    actions = torch.zeros(
        (task.num_envs, task.task_config.action_space_dim), device=task.device
    )
    print("NavRL 3D manual mode: I/K/J/L move, U/O yaw, M toggles control.")
    with torch.no_grad():
        completed = 0
        while completed < int(args.num_trials):
            _, _, terminated, truncated, _ = task.step(actions)
            newly_finished = int(torch.sum(terminated | truncated).item())
            completed += newly_finished
            if newly_finished:
                print(
                    "NavRL 3D manual trial: %d/%d"
                    % (min(completed, int(args.num_trials)), int(args.num_trials))
                )
                if completed < int(args.num_trials):
                    task.reset()
    task.close()


def _run_policy(args):
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit("checkpoint not found: %s" % checkpoint)
    apply_runtime_environment(args)
    os.environ["PLAY_GAMES_NUM"] = str(int(args.num_trials))
    config_file = (
        "ppo_navrl_perception_transformer.yaml"
        if args.policy_kind == "transformer"
        else "ppo_navrl_vision.yaml"
    )
    runner = RL_DIR / "runner.py"
    old_cwd = Path.cwd()
    try:
        os.chdir(RL_DIR)
        sys.argv = [
            str(runner),
            "--file", config_file,
            "--task", "navrl_task",
            "--num_envs", "1",
            "--headless", "False",
            "--use_warp", "True",
            "--play",
            "--checkpoint", str(checkpoint),
        ]
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        os.chdir(old_cwd)


def main():
    args = _setup_dialog(_args())
    if not args.manual and not hasattr(args, "policy_kind"):
        try:
            configure_policy_checkpoint(args)
        except ValueError as exc:
            raise SystemExit("Checkpoint compatibility error: %s" % exc)
    if args.manual:
        _run_manual(args)
    else:
        _run_policy(args)


if __name__ == "__main__":
    main()
