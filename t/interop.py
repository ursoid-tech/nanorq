#!/usr/bin/env python3
"""Check bidirectional RaptorQ interoperability using deterministic fixtures."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
RUST_TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "t/raptorq_interop/target"))
RUST_BIN = RUST_TARGET.resolve() / "release"
TOOLS = {
    "nanorq": ROOT / "t/00util/interop",
    "raptorq": RUST_BIN / "fixture",
}
MAGIC = b"NRQFEC1\0"
HEADER = struct.Struct(">8sQII")
PACKET = struct.Struct(">II")
MASK = (1 << 64) - 1


class Random:
    """Specified SplitMix64 and Fisher-Yates, independent of Python's RNG."""
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.next() % (i + 1)
            items[i], items[j] = items[j], items[i]


def case_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def read_fixture(path):
    data = path.read_bytes()
    magic, common, scheme, count = HEADER.unpack_from(data)
    assert magic == MAGIC, "fixture magic"
    size, symbol_size = common >> 24, (common & 0xffff) + 1
    blocks = (scheme >> 24) + 1
    subblocks, alignment = ((scheme >> 8) & 0xffff) + 1, scheme & 255
    total = (size + symbol_size - 1) // symbol_size
    small, large_count = divmod(total, blocks)
    ks = [small + (sbn < large_count) for sbn in range(blocks)]
    packets = []
    pos = HEADER.size
    for _ in range(count):
        tag, length = PACKET.unpack_from(data, pos)
        pos += PACKET.size
        assert length == symbol_size and pos + length <= len(data), "symbol length"
        assert tag >> 24 < blocks, "source block number"
        packets.append((tag, data[pos:pos + length]))
        pos += length
    assert pos == len(data), "fixture trailing bytes"
    geometry = {"F": size, "T": symbol_size, "Z": blocks, "N": subblocks,
                "Al": alignment, "K": ks}
    return common, scheme, packets, geometry


def write_fixture(path, common, scheme, packets):
    with path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, common, scheme, len(packets)))
        for tag, symbol in packets:
            output.write(PACKET.pack(tag, len(symbol)))
            output.write(symbol)


def command(args, log, expected=0):
    with log.open("wb") as output:
        result = subprocess.run([str(a) for a in args], stdout=output,
                                stderr=subprocess.STDOUT, timeout=300)
    assert result.returncode == expected, (
        f"exit {result.returncode}, expected {expected}: {' '.join(map(str, args))}; "
        f"see {log}")


def decode_case(directory, manifest):
    decoder = manifest["decoder"]
    output = directory / "decoded.bin"
    output.unlink(missing_ok=True)
    command([TOOLS[decoder], "decode", directory / "received.rqf", output],
            directory / "decode.log", manifest["expected_exit"])
    if manifest["expected_exit"] == 0:
        assert output.read_bytes() == (directory / "payload.bin").read_bytes(), "payload mismatch"
    else:
        assert not output.exists(), "incomplete or malformed fixture published output"


def check_cli():
    """Exercise the native-endian data.rq format used by the shipped CLIs."""
    with tempfile.TemporaryDirectory(prefix="nanorq-cli-") as work:
        directory = Path(work)
        payload = hashlib.shake_256(b"cli").digest(131073)
        (directory / "input.bin").write_bytes(payload)
        rust = RUST_BIN / "raptorq_interop"
        for encoder, decoder in [
            ([ROOT / "encode", "input.bin", "1280", "0", "0"],
             [rust, "decode", "data.rq", "output.bin"]),
            ([rust, "encode", "input.bin", "data.rq", "1280", "0", "0"],
             [ROOT / "decode", "output.bin"]),
        ]:
            subprocess.run(encoder, cwd=directory, check=True, timeout=300)
            subprocess.run(decoder, cwd=directory, check=True, timeout=300)
            assert (directory / "output.bin").read_bytes() == payload, "CLI payload mismatch"
            (directory / "output.bin").unlink()
            (directory / "data.rq").unlink()
    print("PASS CLI data.rq (both directions)", flush=True)


def select_packets(packets, ks, repairs, loss, shape, order, seed):
    if shape in ("rank-failure", "rank-recovery"):
        # Known K=10 dependent equation, also covered by native api_regress.c.
        assert ks == [10] and repairs == 133
        by_tag = dict(packets)
        tags = list(range(1, 10)) + [142]
        if shape == "rank-recovery":
            tags.append(10)
        return [(tag, by_tag[tag]) for tag in tags], [{
            "sbn": 0, "K": 10, "sent_repairs": repairs, "lost_sources": 1,
            "lost_repairs": repairs - (len(tags) - 9),
            "surviving_repairs": len(tags) - 9,
        }]
    rng = Random(seed)
    dropped = set()
    budgets = []
    for sbn, k in enumerate(ks):
        source = [i for i, (tag, _) in enumerate(packets)
                  if tag >> 24 == sbn and (tag & 0xffffff) < k]
        repair = [i for i, (tag, _) in enumerate(packets)
                  if tag >> 24 == sbn and (tag & 0xffffff) >= k]
        assert len(source) == k and len(repair) == repairs, "unexpected encoder packet counts"
        lost_repairs = 2 if shape == "repair-loss" else 0
        dropped.update(repair[:lost_repairs])
        if shape == "over-budget":
            lost_sources = min(k, repairs + 1) if sbn == 0 else 0
        elif shape == "single":
            lost_sources = 1
        elif shape == "repair-only":
            lost_sources = k
        elif shape == "repair-loss":
            lost_sources = min(k, repairs - lost_repairs - 2)
        else:
            lost_sources = math.floor(k * loss / 100)
        if shape in ("random", "repair-loss"):
            rng.shuffle(source)
        elif shape == "late":
            source.reverse()
        elif shape == "periodic":
            source = source[::2] + source[1::2]
        dropped.update(source[:lost_sources])
        budgets.append({"sbn": sbn, "K": k, "sent_repairs": repairs,
                        "lost_sources": lost_sources, "lost_repairs": lost_repairs,
                        "surviving_repairs": repairs - lost_repairs})
    if loss > 0 or shape in ("repair-only", "repair-loss", "over-budget", "single"):
        assert sum(b["lost_sources"] for b in budgets) > 0, "case did not require repair"
    if shape == "over-budget":
        assert budgets[0]["lost_sources"] > budgets[0]["surviving_repairs"]
    else:
        assert all(b["lost_sources"] <= b["surviving_repairs"] for b in budgets)
    kept = [i for i in range(len(packets)) if i not in dropped]
    if order in ("shuffle", "duplicates"):
        rng.shuffle(kept)
    elif order == "repair-first":
        kept.sort(key=lambda i: (packets[i][0] & 0xffffff) < ks[packets[i][0] >> 24])
    if order == "duplicates":
        # Repeated source AND repair identifiers must not count as new equations.
        source = next(i for i in kept if (packets[i][0] & 0xffffff) < ks[packets[i][0] >> 24])
        repair = next(i for i in kept if (packets[i][0] & 0xffffff) >= ks[packets[i][0] >> 24])
        kept = [i for index, i in enumerate(kept)
                for _ in range(2 if index % 3 == 0 or i in (source, repair) else 1)]
    return [packets[i] for i in kept], budgets


def check_geometry(manifest, geometry):
    size = manifest["size"]
    symbol_size = manifest["symbol_size"]
    blocks = manifest["blocks"]
    assert geometry["F"] == size, geometry
    if manifest["encoder"] == "raptorq":
        assert geometry["N"] >= manifest.get("rust_subblocks_min", 1), "expected interleaved sub-blocks"
    if blocks:
        total = (size + symbol_size - 1) // symbol_size
        small, large = divmod(total, blocks)
        assert geometry == {"F": size, "T": symbol_size, "Z": blocks,
                            "N": 1, "Al": 8,
                            "K": [small + (i < large) for i in range(blocks)]}, geometry


def run_case(run_dir, seed, name, size, symbol_size, blocks, repairs,
             loss=0, shape="random", order="source-first",
             rust_subblocks_min=1):
    """A fixed repair count is chosen before any packet is removed."""
    success = True
    hashes = []
    for encoder, decoder in (("nanorq", "raptorq"), ("raptorq", "nanorq")):
        directory = run_dir / f"{name}-{encoder}"
        directory.mkdir()
        trace_seed = case_seed(seed, name)
        manifest = {"name": name, "seed": trace_seed, "encoder": encoder, "decoder": decoder,
                    "size": size, "symbol_size": symbol_size, "blocks": blocks,
                    "rust_subblocks_min": rust_subblocks_min,
                    "repairs": repairs, "loss": loss, "shape": shape, "order": order,
                    "expected_exit": 2 if shape in ("over-budget", "rank-failure") else 0}
        try:
            # SHAKE's specified byte stream gives repeatable, nontrivial input.
            payload = hashlib.shake_256(str(trace_seed).encode()).digest(size)
            (directory / "payload.bin").write_bytes(payload)
            del payload
            command([TOOLS[encoder], "encode", directory / "payload.bin",
                     directory / "encoded.rqf", symbol_size, blocks, repairs],
                    directory / "encode.log")
            common, scheme, packets, geometry = read_fixture(directory / "encoded.rqf")
            manifest["geometry"] = geometry
            check_geometry(manifest, geometry)
            hashes.append(hashlib.sha256((directory / "encoded.rqf").read_bytes()).hexdigest())
            received, budgets = select_packets(packets, geometry["K"], repairs,
                                               loss, shape, order, trace_seed)
            manifest["budgets"] = budgets
            manifest["delivered_tags"] = [tag for tag, _ in received]
            write_fixture(directory / "received.rqf", common, scheme, received)
            del packets, received
            (directory / "case.json").write_text(json.dumps(manifest, indent=2) + "\n")
            decode_case(directory, manifest)
        except (AssertionError, OSError, ValueError, struct.error, subprocess.TimeoutExpired) as error:
            success = False
            manifest["failure"] = str(error)
            (directory / "case.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"FAIL {name}/{encoder}: {error}\n  retained: {directory}", flush=True)
    # Equal OTI and symbol ids must produce identical source and repair bytes.
    if blocks and len(hashes) == 2 and hashes[0] != hashes[1]:
        success = False
        print(f"FAIL {name}: encoders produced different fixture bytes", flush=True)
        for encoder, peer in (("nanorq", "raptorq"), ("raptorq", "nanorq")):
            path = run_dir / f"{name}-{encoder}" / "case.json"
            manifest = json.loads(path.read_text())
            manifest["failure"] = "encoders produced different fixture bytes"
            manifest["peer_fixture"] = f"../{name}-{peer}/encoded.rqf"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
    if success:
        for encoder in TOOLS:
            shutil.rmtree(run_dir / f"{name}-{encoder}")
        print(f"PASS {name} (both directions)", flush=True)
    return success


def malformed_cases(run_dir):
    """Reject malformed suffixes even if early packets already decode the object."""
    source = run_dir / "malformed-source"
    source.mkdir()
    (source / "payload.bin").write_bytes(bytes(range(96)))
    command([TOOLS["nanorq"], "encode", source / "payload.bin", source / "encoded.rqf", 96, 1, 0],
            source / "encode.log")
    valid = (source / "encoded.rqf").read_bytes()
    wrong_length = bytearray(valid)
    wrong_length[28:32] = struct.pack(">I", 95)
    wrong_tag = bytearray(valid)
    wrong_tag[24:28] = struct.pack(">I", 1 << 24)
    variants = {"short-header": valid[:20], "short-tag": valid[:26],
                "short-symbol": valid[:-1], "trailing": valid + b"x",
                "symbol-length": wrong_length, "block-number": wrong_tag}
    common = (96 << 24) | 95
    for name, bad_common, bad_scheme, count in [
        ("zero-F", 95, 8, 1),
        ("oversized-F", ((128 * 1024 * 1024 + 1) << 24) | 95, 8, 1),
        ("reserved-bits", common | (1 << 16), 8, 1),
        ("oversized-T", (96 << 24) | 0xffff, 8, 1),
        ("zero-alignment", common, 0, 1),
        ("invalid-alignment", common, 3, 1),
        ("excessive-subblocks", common, (12 << 8) | 8, 1),
        ("excessive-blocks", common, (2 << 24) | 8, 1),
        ("excessive-K", ((56404 * 96) << 24) | 95, 8, 1),
        ("excessive-packets", common, 8, 1000001),
    ]:
        variants[name] = HEADER.pack(MAGIC, bad_common, bad_scheme, count) + valid[24:]
    passed = 0
    for name, data in variants.items():
        for decoder in TOOLS:
            directory = run_dir / f"malformed-{name}-{decoder}"
            directory.mkdir()
            (directory / "received.rqf").write_bytes(data)
            manifest = {"name": name, "decoder": decoder, "expected_exit": 1}
            (directory / "case.json").write_text(json.dumps(manifest, indent=2) + "\n")
            try:
                decode_case(directory, manifest)
                shutil.rmtree(directory)
                passed += 1
            except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
                print(f"FAIL malformed/{name}/{decoder}: {error}", flush=True)
    shutil.rmtree(source)
    print(f"Framing rejection: {passed}/{len(variants) * len(TOOLS)} passed", flush=True)
    return passed == len(variants) * len(TOOLS)


def replay_case(directory):
    manifest = json.loads((directory / "case.json").read_text())
    received_path = directory / "received.rqf"
    with tempfile.TemporaryDirectory(prefix="replay-") as scratch_name:
        scratch = Path(scratch_name)
        if not received_path.exists():
            assert "encoder" in manifest, f"missing {received_path}"
            encoder = manifest["encoder"]
            encoded = scratch / "encoded.rqf"
            command([TOOLS[encoder], "encode", directory / "payload.bin", encoded,
                     manifest["symbol_size"], manifest["blocks"], manifest["repairs"]],
                    scratch / "encode.log")
            common, scheme, packets, geometry = read_fixture(encoded)
            check_geometry(manifest, geometry)
            received, _ = select_packets(
                packets, geometry["K"], manifest["repairs"], manifest["loss"],
                manifest["shape"], manifest["order"], manifest["seed"])
            received_path = scratch / "received.rqf"
            write_fixture(received_path, common, scheme, received)
        if "peer_fixture" in manifest:
            encoder = manifest["encoder"]
            peer = "raptorq" if encoder == "nanorq" else "nanorq"
            reencoded = scratch / f"{encoder}.rqf"
            peer_reencoded = scratch / f"{peer}.rqf"
            for tool, out in ((encoder, reencoded), (peer, peer_reencoded)):
                command([TOOLS[tool], "encode", directory / "payload.bin", out,
                         manifest["symbol_size"], manifest["blocks"], manifest["repairs"]],
                        scratch / f"{tool}-encode.log")
            assert reencoded.read_bytes() == peer_reencoded.read_bytes(), (
                "encoders produced different fixture bytes")
        output = scratch / "decoded.bin"
        command([TOOLS[manifest["decoder"]], "decode", received_path, output],
                directory / "decode.log", manifest["expected_exit"])
        if manifest["expected_exit"] == 0:
            assert output.read_bytes() == (directory / "payload.bin").read_bytes(), "payload mismatch"
        else:
            assert not output.exists(), "incomplete or malformed fixture published output"


def main():
    if not __debug__:
        raise SystemExit("interop checks require Python assertions; unset PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small deterministic CI/development subset")
    parser.add_argument("--seed", type=int, default=1, help="reproducible corpus seed (default: 1)")
    parser.add_argument("--replay", type=Path, help="replay a retained case directory")
    args = parser.parse_args()
    if args.replay:
        directory = args.replay.resolve()
        replay_case(directory)
        print(f"Replay passed: {directory}")
        return 0
    check_cli()
    artifacts = ROOT / "t/interop_test_dir"
    artifacts.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=artifacts))
    print(f"seed={args.seed}; failure artifacts: {run_dir}", flush=True)
    results = []

    def case(*pos, **kw):
        results.append(run_case(run_dir, args.seed, *pos, **kw))

    sizes = [1, 1300, 65536] if args.quick else [1, 16, 1300, 65536, 131072, 1000000]
    symbols = [16, 1280] if args.quick else [16, 496, 1280, 1400]
    rates = [0, 50] if args.quick else [0, 1, 10, 50, 99]
    for size in sizes:
        for symbol in symbols:
            total = (size + symbol - 1) // symbol
            blocks = (total + 4095) // 4096
            max_k = (total + blocks - 1) // blocks
            for rate in rates:
                # Tiny blocks may have no losses after rounding: name that honestly.
                actual_rate = rate if math.floor((total // blocks) * rate / 100) else 0
                repairs = math.ceil(max_k * rate / 100) + 5
                case(f"grid-{size}-{symbol}-{rate}", size, symbol, blocks, repairs, loss=actual_rate)
    for k in ([10, 50, 100] if args.quick else [10, 50, 100, 500, 1000, 5000, 10000, 50000]):
        for rate in ([0, 10] if k >= 5000 else rates):
            actual_rate = rate if math.floor(k * rate / 100) else 0
            case(f"K-{k}-{rate}", k * 1280, 1280, 1, math.ceil(k * rate / 100) + 5, loss=actual_rate)

    # Fixed budgets, including unequal blocks and a partially filled last symbol.
    for shape, order in [("front", "source-first"), ("late", "shuffle"),
                         ("periodic", "repair-first"), ("random", "duplicates"),
                         ("repair-loss", "shuffle"), ("over-budget", "duplicates")]:
        case(f"budget-{shape}-{order}", 31 * 96 - 13, 96, 3, 6,
             loss=20, shape=shape, order=order)
    case("repair-only", 13 * 96 - 7, 96, 1, 18, shape="repair-only", order="shuffle")
    for k in [1, 2, 4, 8]:
        case(f"tiny-repair-{k}", k * 96 - 1, 96, 1, k + 5, shape="repair-only", order="shuffle")
    for shape in ["rank-failure", "rank-recovery"]:
        case(shape, 80, 8, 1, 133, shape=shape)
    # Preserve coverage of each implementation's automatic partition/alignment.
    case("defaults", 65536, 500, 0, 8, loss=0)
    case("defaults-small-symbol", 65536, 16, 0, 6, shape="single")
    if not args.quick:
        case("defaults-interleaved", 10000 * 1280, 1280, 0, 6,
             shape="single", rust_subblocks_min=2)
    results.append(malformed_cases(run_dir))
    failures = results.count(False)
    print(f"Results: {len(results) - failures} groups passed, {failures} failed", flush=True)
    if failures:
        print(f"Failures retained in {run_dir}; replay with --replay CASE_DIRECTORY", flush=True)
    else:
        run_dir.rmdir()
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
