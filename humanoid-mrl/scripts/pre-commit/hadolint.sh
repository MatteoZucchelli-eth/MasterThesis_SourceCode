#!/bin/bash
set -euo pipefail

CACHE_DIR="${HOME}/.cache/pre-commit/hadolint"
BINARY="${CACHE_DIR}/hadolint"
VERSION="v2.14.0"

# Download and cache hadolint binary if missing
if [ ! -f "$BINARY" ]; then
  mkdir -p "$CACHE_DIR"
  echo "Downloading hadolint binary"
  OS=$(uname | tr '[:upper:]' '[:lower:]')
  case "$OS" in
  darwin) OS="macos" ;;
  linux) OS="linux" ;;
  esac
  ARCH=$(uname -m)
  case "$ARCH" in
  x86_64) ARCH="x86_64" ;;
  aarch64) ARCH="arm64" ;;
  arm64) ARCH="arm64" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
  esac

  curl -sL "https://github.com/hadolint/hadolint/releases/download/${VERSION}/hadolint-${OS}-${ARCH}" -o "$BINARY"
  chmod +x "$BINARY"
fi

FILES="$*"

if [ -z "$FILES" ]; then
  FILES=$(git ls-files '*.sh')
fi

if [ -n "$FILES" ]; then
  #shellcheck disable=SC2086
  $BINARY $FILES
fi
