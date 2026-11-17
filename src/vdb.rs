//! Minimal OpenVDB writer: one dense float grid ("density", Tree_float_5_4_3), uncompressed,
//! every voxel inside N³ active. Written to match OpenVDB's own stream layout (file format 224):
//! header, metadata, grid descriptor, then per grid: compression flags, metadata, transform,
//! tree topology, then leaf buffers in the same node order.
//!
//! Voxel (i, j, k) holds the sample at tile position ((i, j, k) + 0.5) / N; VDB places voxel
//! centres at (i, j, k) × voxel size, so the grid sits half a voxel off from the tile.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

const LEAF: u32 = 8; // 2^3
const L4: u32 = 128; // 16 leaves
const L5: u32 = 4096; // 32 L4 nodes
/// "Metadata" byte of a value block: no mask compression, all values follow.
const ALL_VALUES: u8 = 6;

struct W {
    out: BufWriter<File>,
    pos: u64,
}

impl W {
    fn bytes(&mut self, b: &[u8]) -> io::Result<()> {
        self.pos += b.len() as u64;
        self.out.write_all(b)
    }
    fn u32(&mut self, v: u32) -> io::Result<()> {
        self.bytes(&v.to_le_bytes())
    }
    fn i32(&mut self, v: i32) -> io::Result<()> {
        self.bytes(&v.to_le_bytes())
    }
    fn i64(&mut self, v: i64) -> io::Result<()> {
        self.bytes(&v.to_le_bytes())
    }
    fn f64x3(&mut self, v: f64) -> io::Result<()> {
        (0..3).try_for_each(|_| self.bytes(&v.to_le_bytes()))
    }
    fn string(&mut self, s: &str) -> io::Result<()> {
        self.u32(s.len() as u32)?;
        self.bytes(s.as_bytes())
    }
    /// A node mask of `bits` bits, as u64 words.
    fn mask(&mut self, bits: usize, on: impl Fn(usize) -> bool) -> io::Result<()> {
        for word in 0..bits / 64 {
            let mut w = 0u64;
            for b in 0..64 {
                if on(word * 64 + b) {
                    w |= 1 << b;
                }
            }
            self.bytes(&w.to_le_bytes())?;
        }
        Ok(())
    }
    fn zeros(&mut self, count: usize) -> io::Result<()> {
        self.bytes(&[ALL_VALUES])?;
        self.bytes(&vec![0u8; count * 4])
    }
}

/// Origins of the child nodes of size `child` inside a node at `origin` of size `size`,
/// clipped to the grid, in OpenVDB's order (x slowest, z fastest), with their child index.
fn children(origin: [u32; 3], size: u32, child: u32, n: u32) -> Vec<(usize, [u32; 3])> {
    let dim = size / child;
    let mut v = Vec::new();
    for x in 0..dim {
        for y in 0..dim {
            for z in 0..dim {
                let o = [origin[0] + x * child, origin[1] + y * child, origin[2] + z * child];
                if o.iter().all(|&c| c < n) {
                    v.push((((x * dim + y) * dim + z) as usize, o));
                }
            }
        }
    }
    v
}

/// Writes an N³ grid. `eval(origin, size)` returns that box's values, x fastest.
pub fn write(path: &Path, n: u32, mut eval: impl FnMut([u32; 3], [u32; 3]) -> Vec<f32>, mut progress: impl FnMut(f32)) -> io::Result<()> {
    let mut w = W { out: BufWriter::new(File::create(path)?), pos: 0 };

    // File header.
    w.i64(0x5644_4220)?; // magic "VDB "
    w.u32(224)?; // file format version
    w.u32(10)?; // library major
    w.u32(0)?; // library minor
    w.bytes(&[1])?; // has grid offsets
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let h = format!("{:032x}", t ^ 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C834);
    w.bytes(format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32]).as_bytes())?; // uuid
    w.i32(0)?; // file metadata count
    w.i32(1)?; // grid count

    // Grid descriptor; its three offsets are filled in at the end.
    w.string("density")?;
    w.string("Tree_float_5_4_3")?;
    w.string("")?; // instance parent
    let offsets_at = w.pos;
    for _ in 0..3 {
        w.i64(0)?;
    }
    let grid_pos = w.pos;
    w.u32(0)?; // compression: none
    // Grid metadata: OpenVDB reads the grid's name from here, not from the descriptor.
    w.i32(1)?;
    w.string("name")?;
    w.string("string")?;
    w.string("density")?; // string metadata value: length + bytes
    let s = 1.0 / n as f64;
    w.string("UniformScaleMap")?;
    w.f64x3(s)?; // scale
    w.f64x3(s)?; // voxel size
    w.f64x3(1.0 / s)?; // inverse scale
    w.f64x3(1.0 / (s * s))?; // inverse scale squared
    w.f64x3(1.0 / (2.0 * s))?; // inverse twice scale

    // Topology.
    let roots = children([0; 3], L5 * n.div_ceil(L5), L5, n);
    w.i32(1)?; // buffer count
    w.bytes(&0f32.to_le_bytes())?; // background
    w.u32(0)?; // tiles
    w.u32(roots.len() as u32)?;
    let inside = |o: [u32; 3], i: usize| {
        let (x, y, z) = (o[0] + (i as u32 >> 6), o[1] + ((i as u32 >> 3) & 7), o[2] + (i as u32 & 7));
        x < n && y < n && z < n
    };
    for &(_, o5) in &roots {
        (0..3).try_for_each(|a| w.i32(o5[a] as i32))?;
        let l4s = children(o5, L5, L4, n);
        w.mask(32 * 32 * 32, |i| l4s.iter().any(|c| c.0 == i))?; // child mask
        w.mask(32 * 32 * 32, |_| false)?; // value (tile) mask
        w.zeros(32 * 32 * 32)?;
        for &(_, o4) in &l4s {
            let leaves = children(o4, L4, LEAF, n);
            w.mask(16 * 16 * 16, |i| leaves.iter().any(|c| c.0 == i))?;
            w.mask(16 * 16 * 16, |_| false)?;
            w.zeros(16 * 16 * 16)?;
            for &(_, ol) in &leaves {
                w.mask(512, |i| inside(ol, i))?;
            }
        }
    }

    // Leaf buffers, one L4 node evaluated at a time.
    let block_pos = w.pos;
    let total: usize = roots.iter().map(|&(_, o5)| children(o5, L5, L4, n).len()).sum();
    let mut done = 0;
    for &(_, o5) in &roots {
        for (_, o4) in children(o5, L5, L4, n) {
            let size = [L4.min(n - o4[0]), L4.min(n - o4[1]), L4.min(n - o4[2])];
            let v = eval(o4, size);
            for (_, ol) in children(o4, L4, LEAF, n) {
                w.mask(512, |i| inside(ol, i))?;
                w.bytes(&[ALL_VALUES])?;
                let mut vals = [0u8; 512 * 4];
                for i in 0..512usize {
                    if inside(ol, i) {
                        let (x, y, z) = (ol[0] - o4[0] + (i as u32 >> 6), ol[1] - o4[1] + ((i as u32 >> 3) & 7), ol[2] - o4[2] + (i as u32 & 7));
                        let src = ((z * size[1] + y) * size[0] + x) as usize;
                        vals[i * 4..i * 4 + 4].copy_from_slice(&v[src].to_le_bytes());
                    }
                }
                w.bytes(&vals)?;
            }
            done += 1;
            progress(done as f32 / total as f32);
        }
    }
    let end_pos = w.pos;

    w.out.seek(SeekFrom::Start(offsets_at))?;
    for p in [grid_pos, block_pos, end_pos] {
        w.out.write_all(&(p as i64).to_le_bytes())?;
    }
    w.out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_follow_openvdb_order_and_clip() {
        // A 20³ grid inside one L4 node: leaves at 0, 8, 16 per axis, x slowest.
        let c = children([0; 3], L4, LEAF, 20);
        assert_eq!(c.len(), 27);
        assert_eq!(c[0], (0, [0, 0, 0]));
        assert_eq!(c[1], (1, [0, 0, 8])); // z fastest
        assert_eq!(c[3], (16, [0, 8, 0]));
        assert_eq!(c[9], (256, [8, 0, 0]));
    }

    #[test]
    fn writes_expected_size() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("test.vdb");
        write(&path, 16, |_, s| vec![0.5; (s[0] * s[1] * s[2]) as usize], |_| {}).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], &0x5644_4220i64.to_le_bytes());
        // 8 leaves × (64-byte mask + 1 + 2048) in the buffer section alone.
        assert!(bytes.len() > 8 * (64 + 1 + 2048));
    }
}
