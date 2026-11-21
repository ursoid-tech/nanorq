#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi
make -C "$ROOT" encode decode t/00util/interop
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/t/raptorq_interop/target}"
cargo test --manifest-path "$ROOT/t/raptorq_interop/Cargo.toml" --locked --release --bin fixture
cargo build --manifest-path "$ROOT/t/raptorq_interop/Cargo.toml" --locked --release --bin fixture --bin raptorq_interop
exec python3 "$ROOT/t/interop.py" "$@"
