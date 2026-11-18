//! PLY and STL readers, so meshes from Houdini, ZBrush, Blender and CAD tools can be used without
//! converting them to OBJ first. Only the geometry is read (positions and faces, with polygons
//! fanned into triangles).

use crate::mesh::{Mesh, V3};
use std::path::Path;

/// Reads a mesh by file extension: obj, ply or stl.
pub fn load(path: &Path) -> Result<Mesh, String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "ply" => load_ply(path),
        "stl" => load_stl(path),
        _ => Mesh::load_obj(path),
    }
}

/// Whether a file name has an extension `load` understands.
pub fn is_mesh_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| ["obj", "ply", "stl"].iter().any(|x| e.eq_ignore_ascii_case(x)))
}

fn finish(mesh: Mesh, path: &Path) -> Result<Mesh, String> {
    if mesh.triangles.is_empty() {
        return Err(format!("{} has no triangles", path.display()));
    }
    Ok(mesh.welded())
}

// ---- STL -------------------------------------------------------------------

fn load_stl(path: &Path) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let mut mesh = Mesh::default();
    // Binary: 80 header bytes, a u32 triangle count, then 50 bytes per triangle. An ASCII file also
    // starts with "solid", so the size decides.
    let binary = bytes.len() >= 84 && {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        bytes.len() == 84 + 50 * n
    };
    if binary {
        for t in bytes[84..].chunks_exact(50) {
            let base = mesh.positions.len() as u32;
            for v in 0..3 {
                let o = 12 + v * 12;
                let g = |k: usize| f32::from_le_bytes([t[o + 4 * k], t[o + 4 * k + 1], t[o + 4 * k + 2], t[o + 4 * k + 3]]);
                mesh.positions.push([g(0), g(1), g(2)]);
            }
            mesh.triangles.push([base, base + 1, base + 2]);
        }
    } else {
        let text = String::from_utf8_lossy(&bytes);
        let mut corner = Vec::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            if w.next() == Some("vertex") {
                let v: Vec<f32> = w.filter_map(|x| x.parse().ok()).collect();
                if v.len() == 3 {
                    corner.push([v[0], v[1], v[2]]);
                }
            }
        }
        for tri in corner.chunks_exact(3) {
            let base = mesh.positions.len() as u32;
            mesh.positions.extend_from_slice(tri);
            mesh.triangles.push([base, base + 1, base + 2]);
        }
    }
    finish(mesh, path)
}

// ---- PLY -------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    Little,
    Big,
}

#[derive(Clone, Copy)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Option<Ty> {
        Some(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
}

struct Property {
    name: String,
    ty: Ty,
    /// For list properties, the type of the count.
    count: Option<Ty>,
}

struct Element {
    name: String,
    n: usize,
    props: Vec<Property>,
}

/// Reads scalars from the body in the file's format.
struct Body<'a> {
    data: &'a [u8],
    pos: usize,
    format: Format,
}

impl Body<'_> {
    fn next(&mut self, ty: Ty) -> Result<f64, String> {
        if self.format == Format::Ascii {
            while self.pos < self.data.len() && self.data[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            let start = self.pos;
            while self.pos < self.data.len() && !self.data[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            return std::str::from_utf8(&self.data[start..self.pos]).ok().and_then(|s| s.parse().ok()).ok_or("bad number in the PLY body".to_string());
        }
        let n = ty.size();
        let bytes = self.data.get(self.pos..self.pos + n).ok_or("the PLY file ends early")?;
        self.pos += n;
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(bytes);
        if self.format == Format::Big {
            b[..n].reverse();
        }
        Ok(match ty {
            Ty::I8 => b[0] as i8 as f64,
            Ty::U8 => b[0] as f64,
            Ty::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Ty::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            Ty::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Ty::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Ty::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Ty::F64 => f64::from_le_bytes(b),
        })
    }
}

fn load_ply(path: &Path) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let end = bytes.windows(10).position(|w| w == b"end_header").ok_or("not a PLY file (no end_header)")?;
    let header = String::from_utf8_lossy(&bytes[..end]).into_owned();
    let mut body_start = end + 10;
    while body_start < bytes.len() && bytes[body_start] != b'\n' {
        body_start += 1;
    }
    body_start += 1; // past the newline that ends the header
    let mut lines = header.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err("not a PLY file".into());
    }
    let (mut format, mut elements) = (Format::Ascii, Vec::<Element>::new());
    for line in lines {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.as_slice() {
            ["format", f, ..] => {
                format = match *f {
                    "ascii" => Format::Ascii,
                    "binary_little_endian" => Format::Little,
                    "binary_big_endian" => Format::Big,
                    other => return Err(format!("unknown PLY format {other}")),
                }
            }
            ["element", name, n] => elements.push(Element { name: name.to_string(), n: n.parse().map_err(|_| "bad element count")?, props: Vec::new() }),
            ["property", "list", count, ty, name] => {
                let (count, ty) = (Ty::parse(count).ok_or("unknown PLY type")?, Ty::parse(ty).ok_or("unknown PLY type")?);
                elements.last_mut().ok_or("property before element")?.props.push(Property { name: name.to_string(), ty, count: Some(count) });
            }
            ["property", ty, name] => {
                let ty = Ty::parse(ty).ok_or_else(|| format!("unknown PLY type {ty}"))?;
                elements.last_mut().ok_or("property before element")?.props.push(Property { name: name.to_string(), ty, count: None });
            }
            _ => {}
        }
    }
    let mut body = Body { data: &bytes, pos: body_start, format };
    let mut mesh = Mesh::default();
    for el in &elements {
        for _ in 0..el.n {
            let mut xyz: [f64; 3] = [0.0; 3];
            let mut rgb: [Option<f64>; 3] = [None; 3];
            let mut face: Option<Vec<u32>> = None;
            for p in &el.props {
                if let Some(ct) = p.count {
                    let count = body.next(ct)? as usize;
                    let idx = (0..count).map(|_| body.next(p.ty).map(|v| v as u32)).collect::<Result<Vec<_>, _>>()?;
                    if p.name == "vertex_indices" || p.name == "vertex_index" {
                        face = Some(idx);
                    }
                } else {
                    let v = body.next(p.ty)?;
                    // Colours are stored as bytes (0 to 255) or as floats (0 to 1).
                    let scaled = if matches!(p.ty, Ty::U8 | Ty::U16) { v / if matches!(p.ty, Ty::U8) { 255.0 } else { 65535.0 } } else { v };
                    match p.name.as_str() {
                        "x" => xyz[0] = v,
                        "y" => xyz[1] = v,
                        "z" => xyz[2] = v,
                        "red" | "r" | "diffuse_red" => rgb[0] = Some(scaled),
                        "green" | "g" | "diffuse_green" => rgb[1] = Some(scaled),
                        "blue" | "b" | "diffuse_blue" => rgb[2] = Some(scaled),
                        _ => {}
                    }
                }
            }
            if el.name == "vertex" {
                mesh.positions.push([xyz[0] as f32, xyz[1] as f32, xyz[2] as f32] as V3);
                if let [Some(r), Some(g), Some(b)] = rgb {
                    mesh.colors.push([r as f32, g as f32, b as f32]);
                }
            } else if let (true, Some(f)) = (el.name == "face", face) {
                // Fan polygons into triangles.
                for k in 1..f.len().saturating_sub(1) {
                    mesh.triangles.push([f[0], f[k], f[k + 1]]);
                }
            }
        }
    }
    if mesh.colors.len() != mesh.positions.len() {
        mesh.colors.clear(); // not every vertex had a colour
    }
    if mesh.triangles.iter().any(|t| t.iter().any(|&i| i as usize >= mesh.positions.len())) {
        return Err("the PLY file has a face that points past its vertices".into());
    }
    finish(mesh, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::shapes;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("noiser-format-test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn ply_written_by_noiser_reads_back() {
        let ico = shapes::icosphere(1.0, 2);
        let raw = crate::surface::RawMesh { normals: vec![[0.0, 0.0, 1.0]; ico.positions.len()], positions: ico.positions.clone(), triangles: ico.triangles.clone() };
        let path = tmp("ico.ply");
        raw.write_ply(&path).unwrap();
        let back = load(&path).unwrap();
        assert_eq!((back.positions.len(), back.triangles.len()), (ico.positions.len(), ico.triangles.len()));
    }

    #[test]
    fn ply_vertex_colours_are_read_and_scaled() {
        let path = tmp("colour.ply");
        std::fs::write(&path, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 128\n3 0 1 2\n").unwrap();
        let m = load(&path).unwrap();
        assert_eq!(m.colors.len(), 3);
        assert_eq!(m.colors[0], [1.0, 0.0, 0.0]);
        assert!((m.colors[2][2] - 128.0 / 255.0).abs() < 1e-6);
        // A file without colours has none.
        let plain = tmp("plain.ply");
        std::fs::write(&plain, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n").unwrap();
        assert!(load(&plain).unwrap().colors.is_empty());
    }

    #[test]
    fn ascii_ply_with_a_quad_and_binary_stl_read() {
        let ply = tmp("quad.ply");
        std::fs::write(&ply, "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n").unwrap();
        let m = load(&ply).unwrap();
        assert_eq!((m.positions.len(), m.triangles.len()), (4, 2), "a quad is fanned into two triangles");
        // One binary STL triangle.
        let mut stl = vec![0u8; 80];
        stl.extend(1u32.to_le_bytes());
        for v in [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            stl.extend(v.to_le_bytes());
        }
        stl.extend([0u8, 0]);
        let path = tmp("tri.stl");
        std::fs::write(&path, stl).unwrap();
        assert_eq!(load(&path).unwrap().triangles.len(), 1);
        assert!(load(&tmp("missing.ply")).is_err());
        std::fs::write(tmp("bad.ply"), "hello").unwrap();
        assert!(load(&tmp("bad.ply")).is_err());
    }
}
