# nanorq
[![CI](https://github.com/sleepybishop/nanorq/actions/workflows/ci.yml/badge.svg?branch=stable)](https://github.com/sleepybishop/nanorq/actions/workflows/ci.yml)

nanorq is a compact, performant implementation of the raptorq fountain code capable of reaching multi-gigabit speeds on a single core.

## API
  - High-level API `nanorq.h` abstracts allocation, alignment, and I/O handling from the user.
  - Low-level API `nanorq_core.h` provides more flexibility and can be used without any libc dependency[^1], but requires more integration from the user.

The core API also offers an XOR-only profile. Use
`nanorq_core_encoder_new_gf2(K, overhead, &rq)` on both the encoder and the
decoder. This profile omits HDPC rows and produces repair symbols that are
incompatible with RFC 6330. Agree on the profile out of band. If the decoder
receives dependent repairs, it needs additional independent symbols to recover
the block.

## Performance[^2]
![](graph.png)

`make bench-latency` reports p50, p95, and p99 encode, decode, and total host
latency for blocks of 1 to 64 source symbols. Each iteration drops one source
symbol, sends four repairs, and verifies the recovered bytes. The CSV includes
both freshly allocated and reused codec objects. To change the sample count,
build with `make benchmark_latency` and run, for example, `./benchmark_latency 100`.

## Tests

Run `make check` for the native suite. With Python 3 and Cargo available,
`make check-interop` tests encoding and decoding against the Rust `raptorq`
crate in both directions. Use `make check-interop-quick` for the smaller corpus.
To replay a retained fixture, pass its case directory to
`./t/interop_harness.sh --replay CASE_DIRECTORY`.

## Use cases
  - firmware deployment / software updates
  - video streaming
  - large data transfers across high latency links

## Helper Modules

### Tunable Sparse Network Coding (TSNC)
The high-level API includes experimental block-based recoding helpers that
leverage the optimized primitives used for `nanorq`. These helpers are demos,
not part of the RFC 6330 interoperability surface. Dense TSNC recovery is
bounded to 256 MiB of coefficient and payload matrix memory and fails cleanly
when that limit would be exceeded.

Some examples are provided:
  - Blockchain gossip propagation: Accelerating block synchronization across peer-to-peer gossip networks.
  - Multipath recoding: Re-encoding packets at intermediate network hops without full block decoding to maximize throughput.
  - Real-time continuous streaming: Utilizing a sliding window mechanism to recover missing stream data with minimal overhead and latency.

`nanorq` objects cache mutable encoder/decoder state. A single object must not
be used concurrently without external synchronization; separate objects may be
used by separate threads.

[^1]: When compiled with `NDEBUG`. Test suite requires stdlib.
[^2]: Build autodetects and optimizes for the host CPU architecture (using -march=native).
