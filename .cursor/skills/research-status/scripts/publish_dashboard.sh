#!/usr/bin/env bash
# Publish the MOTAR research-status dashboard via GitHub Pages, served straight
# from the research branch's /docs folder (NO separate gh-pages branch).
#
# One-time GitHub setting (repo Settings -> Pages):
#   Source = "Deploy from a branch",  Branch = research/navrl-env,  Folder = /docs
# Then the dashboard is live at:
#   https://joshualikaist.github.io/MOTAR/status/
#
# This script commits ONLY docs/ (dashboard + status.json) on the current branch
# and pushes. It does not touch any other uncommitted work.
#
# Usage: scripts/publish_dashboard.sh [REPO_ROOT]
set -euo pipefail

# Derive the repo from this script's location
# (<repo>/.cursor/skills/research-status/scripts/), so the skill works in any
# clone -- including Cursor cloud/mobile agents, where $HOME differs. The
# workspace-root copy is a symlink into the repo, so resolve it first.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
DERIVED="$(cd "$(dirname "$SELF")/../../../.." && pwd)"
REPO="${1:-$DERIVED}"
if [ ! -f "$REPO/docs/status/index.html" ] && [ -z "${1:-}" ]; then
  LEGACY="$HOME/workspaces/aerial_gym_ws/src/aerial_gym_simulator"
  [ -f "$LEGACY/docs/status/index.html" ] && REPO="$LEGACY"
fi
cd "$REPO"

[ -f docs/status/index.html ] || { echo "error: docs/status/index.html missing (build the dashboard first)"; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git add docs
if git diff --cached --quiet -- docs; then
  echo "no dashboard changes to publish"
  exit 0
fi

# Use -F (not -m) so Cursor/git wrappers that inject unsupported --trailer
# options cannot break commit on older git (e.g. 2.25).
MSG="$(mktemp)"
trap 'rm -f "$MSG"' EXIT
printf 'status: update research dashboard (%s)\n' "$(date -u +%Y-%m-%dT%H:%MZ)" > "$MSG"
git commit -F "$MSG"
# Pages source is research/navrl-env:/docs. Pushing main and that ref in one
# receive does not create a Pages build — source ref must go out alone first.
git push origin HEAD:research/navrl-env
if [ "$BRANCH" != "research/navrl-env" ]; then
  git push origin "$BRANCH"
fi
echo "pushed docs/ to origin/research/navrl-env (Pages source) then origin/$BRANCH"
echo "live (after Pages is enabled on research/navrl-env:/docs): https://joshualikaist.github.io/MOTAR/status/"
