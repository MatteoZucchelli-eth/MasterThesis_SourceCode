# Activate the control container environment and check required modules.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set +u
[ -f /root/venv/bin/activate ] || {
    echo "Run this tool inside the control container: make control" >&2
    exit 1
}
. /root/venv/bin/activate
for _setup in /opt/ros/*/setup.bash "$REPO/install/setup.bash"; do
    [ -f "$_setup" ] && . "$_setup" >/dev/null 2>&1
done
unset _setup
set -u

cd "$REPO"

for _mod in ${REQUIRE:-pandas}; do
    python -c "import $_mod" 2>/dev/null && continue
    echo "Missing Python module '$_mod' in the control container." >&2
    exit 1
done
unset _mod
