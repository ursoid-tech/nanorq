//! Portable packet fixtures for the bidirectional interoperability harness.
//! The existing command-line example's native-endian format is unchanged.

use raptorq::{Decoder, Encoder, EncodingPacket, ObjectTransmissionInformation, PayloadId};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"NRQFEC1\0";
const MAX_PAYLOAD: u64 = 128 * 1024 * 1024;
const MAX_FIXTURE: u64 = 256 * 1024 * 1024;
const MAX_PACKETS: usize = 1_000_000;
const MAX_K: u64 = 56_403;

type Result<T> = std::result::Result<T, String>;

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("input exceeds the test helper size limit".into());
    }
    Ok(bytes)
}

fn common_oti(oti: ObjectTransmissionInformation) -> u64 {
    (oti.transfer_length() << 24) | u64::from(oti.symbol_size() - 1)
}

fn scheme_oti(oti: ObjectTransmissionInformation) -> u32 {
    (u32::from(oti.source_blocks() - 1) << 24)
        | (u32::from(oti.sub_blocks() - 1) << 8)
        | u32::from(oti.symbol_alignment())
}

fn validate_oti(common: u64, scheme: u32) -> Result<ObjectTransmissionInformation> {
    let length = common >> 24;
    let symbol_size = (common & 0xffff) + 1;
    let blocks = u64::from((scheme >> 24) + 1);
    let sub_blocks = u64::from(((scheme >> 8) & 0xffff) + 1);
    let alignment = u64::from(scheme & 0xff);
    if length == 0 || length > MAX_PAYLOAD || common & 0xff0000 != 0 {
        return Err("invalid transfer length or reserved common OTI bits".into());
    }
    if symbol_size > u16::MAX as u64
        || !matches!(alignment, 1 | 2 | 4 | 8)
        || symbol_size % alignment != 0
        || sub_blocks > symbol_size / alignment
        || sub_blocks > u16::MAX as u64
    {
        return Err("invalid symbol size, alignment, or sub-block count".into());
    }
    let symbols = length.div_ceil(symbol_size);
    // Rust raptorq's OTI represents Z as u8; 256 cannot be represented.
    if blocks > u8::MAX as u64 || blocks > symbols || symbols.div_ceil(blocks) > MAX_K {
        return Err("invalid or unsupported source-block geometry".into());
    }
    Ok(ObjectTransmissionInformation::new(
        length,
        symbol_size as u16,
        blocks as u8,
        sub_blocks as u16,
        alignment as u8,
    ))
}

struct Cursor<'a> {
    remaining: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        if length > self.remaining.len() {
            return Err("truncated fixture".into());
        }
        let (value, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
}

struct Fixture<'a> {
    oti: ObjectTransmissionInformation,
    packets: Vec<(u32, &'a [u8])>,
}

fn parse_fixture(bytes: &[u8]) -> Result<Fixture<'_>> {
    if bytes.len() as u64 > MAX_FIXTURE {
        return Err("fixture exceeds the test helper size limit".into());
    }
    let mut input = Cursor { remaining: bytes };
    if input.take(8)? != MAGIC {
        return Err("invalid fixture magic".into());
    }
    let common = input.u64()?;
    let scheme = input.u32()?;
    let oti = validate_oti(common, scheme)?;
    let count = input.u32()? as usize;
    if count > MAX_PACKETS {
        return Err("too many fixture packets".into());
    }
    let record_size = usize::from(oti.symbol_size()) + 8;
    if count.checked_mul(record_size) != Some(input.remaining.len()) {
        return Err("truncated fixture or trailing bytes".into());
    }
    let mut packets = Vec::with_capacity(count);
    for _ in 0..count {
        let tag = input.u32()?;
        let length = input.u32()? as usize;
        if tag >> 24 >= u32::from(oti.source_blocks()) {
            return Err("packet source-block number is out of range".into());
        }
        if length != usize::from(oti.symbol_size()) {
            return Err("packet symbol length does not match OTI".into());
        }
        packets.push((tag, input.take(length)?));
    }
    Ok(Fixture { oti, packets })
}

fn print_geometry(oti: ObjectTransmissionInformation) {
    let symbols = oti.transfer_length().div_ceil(u64::from(oti.symbol_size()));
    let blocks = u64::from(oti.source_blocks());
    eprintln!(
        "F={} T={} Z={} N={} Al={} K_max={} K_min={}",
        oti.transfer_length(),
        oti.symbol_size(),
        oti.source_blocks(),
        oti.sub_blocks(),
        oti.symbol_alignment(),
        symbols.div_ceil(blocks),
        symbols / blocks,
    );
}

fn encode(args: &[String]) -> Result<()> {
    let data = read_bounded(Path::new(&args[0]), MAX_PAYLOAD)?;
    let symbol_size: u16 = args[2].parse().map_err(|_| "invalid symbol size")?;
    let blocks: u8 = args[3].parse().map_err(|_| "invalid source-block count")?;
    let repair: u32 = args[4].parse().map_err(|_| "invalid repair count")?;
    if data.is_empty() || symbol_size == 0 {
        return Err("payload and symbol size must be nonzero".into());
    }
    let oti = if blocks == 0 {
        ObjectTransmissionInformation::with_defaults(data.len() as u64, symbol_size)
    } else {
        if symbol_size % 8 != 0 {
            return Err("explicit block geometry requires symbol size divisible by 8".into());
        }
        validate_oti(
            ((data.len() as u64) << 24) | u64::from(symbol_size - 1),
            (u32::from(blocks - 1) << 24) | 8,
        )?
    };
    // Validate defaults too, before creating the potentially large encoder.
    if oti.source_blocks() == 0 || oti.sub_blocks() == 0 || oti.symbol_size() == 0 {
        return Err("default configuration is not representable".into());
    }
    let oti = validate_oti(common_oti(oti), scheme_oti(oti))?;
    let symbols = oti.transfer_length().div_ceil(u64::from(oti.symbol_size()));
    let packet_count = symbols + u64::from(oti.source_blocks()) * u64::from(repair);
    if packet_count > MAX_PACKETS as u64
        || 24 + packet_count * (8 + u64::from(oti.symbol_size())) > MAX_FIXTURE
        || symbols.div_ceil(u64::from(oti.source_blocks())) + u64::from(repair) > 1 << 24
    {
        return Err("repair count exceeds fixture or encoding-symbol limits".into());
    }
    print_geometry(oti);
    let encoder = Encoder::new(&data, oti);
    let mut out = BufWriter::new(File::create(&args[1]).map_err(|e| e.to_string())?);
    let write =
        |out: &mut BufWriter<File>, bytes: &[u8]| out.write_all(bytes).map_err(|e| e.to_string());
    write(&mut out, MAGIC)?;
    write(&mut out, &common_oti(oti).to_be_bytes())?;
    write(&mut out, &scheme_oti(oti).to_be_bytes())?;
    write(&mut out, &(packet_count as u32).to_be_bytes())?;
    for block in encoder.get_block_encoders() {
        for packet in block
            .source_packets()
            .into_iter()
            .chain(block.repair_packets(0, repair))
        {
            write(&mut out, &packet.payload_id().serialize())?;
            write(&mut out, &(packet.data().len() as u32).to_be_bytes())?;
            write(&mut out, packet.data())?;
        }
    }
    out.flush().map_err(|e| e.to_string())
}

fn decode(args: &[String]) -> Result<u8> {
    let bytes = read_bounded(Path::new(&args[0]), MAX_FIXTURE)?;
    // Validate the entire fixture even if its first few packets suffice to decode.
    let fixture = parse_fixture(&bytes)?;
    print_geometry(fixture.oti);
    let mut decoder = Decoder::new(fixture.oti);
    for (tag, data) in fixture.packets {
        if let Some(decoded) = decoder.decode(EncodingPacket::new(
            PayloadId::deserialize(&tag.to_be_bytes()),
            data.to_vec(),
        )) {
            fs::write(&args[1], decoded).map_err(|e| e.to_string())?;
            return Ok(0);
        }
    }
    eprintln!("fixture is valid but decoding is incomplete");
    Ok(2)
}

fn run() -> Result<u8> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("encode") if args.len() == 7 => {
            encode(&args[2..])?;
            Ok(0)
        }
        Some("decode") if args.len() == 4 => decode(&args[2..]),
        _ => Err("usage: fixture encode <payload> <fixture> <symbol-size> <blocks> <repair-count>\n       fixture decode <fixture> <output>".into()),
    }
}

fn main() {
    std::process::exit(match run() {
        Ok(code) => i32::from(code),
        Err(error) => {
            eprintln!("fixture: {error}");
            1
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_packet() -> Vec<u8> {
        // F=8, T=8, Z=1, N=1, Al=8; one systematic packet with ESI=0.
        let mut bytes =
            b"NRQFEC1\0\x00\x00\x00\x00\x08\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\x01".to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 8]);
        bytes.extend_from_slice(b"payload!");
        bytes
    }

    #[test]
    fn reads_known_big_endian_fixture() {
        let bytes = one_packet();
        let fixture = parse_fixture(&bytes).unwrap();
        assert_eq!(fixture.oti.transfer_length(), 8);
        assert_eq!(fixture.oti.symbol_size(), 8);
        assert_eq!(fixture.packets, vec![(0, b"payload!".as_slice())]);
    }

    #[test]
    fn rejects_truncation_and_trailing_data() {
        let mut bytes = one_packet();
        for end in 0..bytes.len() {
            assert!(parse_fixture(&bytes[..end]).is_err());
        }
        bytes.push(0);
        assert!(parse_fixture(&bytes).is_err());
    }

    #[test]
    fn rejects_invalid_geometry_and_packet_metadata() {
        for (offset, value) in [(19, 0), (16, 255), (13, 1), (24, 1), (31, 7)] {
            let mut bytes = one_packet();
            bytes[offset] = value;
            assert!(parse_fixture(&bytes).is_err(), "offset {offset}");
        }
    }
}
