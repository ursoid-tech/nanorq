#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi
make -C "$ROOT" t/00util/interop
cargo test --manifest-path "$ROOT/t/raptorq_interop/Cargo.toml" --locked --release --bin fixture
cargo build --manifest-path "$ROOT/t/raptorq_interop/Cargo.toml" --locked --release --bin fixture
exec python3 "$ROOT/t/interop.py" "$@"
