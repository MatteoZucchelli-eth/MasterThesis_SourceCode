#!/usr/bin/env bash
# Give the whole repository back to the host user.
#
# Files written from inside a container end up owned by root; this resets them.
#
#   scripts/fix_ownership.sh
set -euo pipefail

OWNER="637651:1029"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

chown -R "${OWNER}" "${REPO_ROOT}"