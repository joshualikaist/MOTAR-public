#!/bin/sh
# NavRL 3-D simulator entry point.
#   ./launch_navrl_3d.sh
#       -> GTK setup window (system Python), then Isaac Gym viewer (aerialgym conda env)
#   ./launch_navrl_3d.sh --checkpoint path/to/model.pth
#   ./launch_navrl_3d.sh --manual
# Optional: --num-trials N --density-min A --density-max B --target-speed V --drone-speed V
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$APP_DIR"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${APP_DIR}"
PY="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHONNOUSERSITE=1
PY_DIR=$(dirname -- "$PY")
export PATH="$PY_DIR:$PATH"
if [ "$#" -eq 0 ] && /usr/bin/python3 -c "import gi; gi.require_version('Gtk', '3.0')" 2>/dev/null; then
    exec /usr/bin/python3 aerial_gym/apps/navrl_3d_gtk_launcher.py --runtime-python "$PY"
fi
exec "${PY}" aerial_gym/apps/navrl_3d.py "$@"
