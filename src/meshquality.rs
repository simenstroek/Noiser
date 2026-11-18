//! Objective numbers for a triangle mesh: triangle shape, smoothness between neighbours, topology, and (given a
//! way to evaluate the true field) how far it lies from the surface it is meant to describe. Every remeshing
//! method is scored with the same report, so they can be compared without looking at them.

use crate::mesh::V3;
use crate::surface::RawMesh;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub vertices: usize,
    pub triangles: usize,
    /// The smallest angle of any triangle, and the median over triangles of each one's smallest angle, in degrees.
    pub min_angle: f32,
    pub median_min_angle: f32,
    /// Fraction of triangles whose smallest angle is below 20 degrees (thin ones).
    pub thin_fraction: f32,
    /// Radius-ratio quality 4 sqrt(3) A / (a² + b² + c²): 1 for an equilateral triangle, towards 0 for slivers.
    pub mean_quality: f32,
    pub worst_quality: f32,
    /// Angle between the normals of neighbouring triangles, in degrees: 95th percentile and maximum.
    pub p95_normal_deviation: f32,
    pub max_normal_deviation: f32,
    /// Edges used by one triangle, and by more than two.
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub components: usize,
    /// Distance from the true surface (set by `with_surface_error`), in the mesh's units.
    pub mean_error: Option<f32>,
    pub max_error: Option<f32>,
}

impl Report {
    pub fn watertight(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} triangles, {} vertices, {} part(s){}", self.triangles, self.vertices, self.components, if self.watertight() { ", watertight" } else { "" })?;
        writeln!(f, "triangle angle: smallest {:.1}°, median of smallest {:.1}°, {:.1}% under 20°", self.min_angle, self.median_min_angle, 100.0 * self.thin_fraction)?;
        writeln!(f, "triangle quality (1 = equilateral): mean {:.2}, worst {:.2}", self.mean_quality, self.worst_quality)?;
        write!(f, "neighbour normal deviation: 95th percentile {:.1}°, max {:.1}°", self.p95_normal_deviation, self.max_normal_deviation)?;
        if self.boundary_edges + self.non_manifold_edges > 0 {
            write!(f, "\nopen edges {}, non-manifold edges {}", self.boundary_edges, self.non_manifold_edges)?;
        }
        if let (Some(mean), Some(max)) = (self.mean_error, self.max_error) {
            write!(f, "\ndistance from the true surface: mean {mean:.5}, max {max:.5}")?;
        }
        Ok(())
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V3) -> f32 {
    dot(a, a).sqrt()
}

/// The `q` quantile (0 to 1) of `values`, which it sorts.
fn quantile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[((values.len() - 1) as f32 * q).round() as usize]
}

/// Triangle shape, neighbour smoothness and topology of `mesh`.
pub fn report(mesh: &RawMesh) -> Report {
    let mut r = Report { vertices: mesh.positions.len(), triangles: mesh.triangles.len(), ..Default::default() };
    if mesh.triangles.is_empty() {
        return r;
    }
    let (mut min_angles, mut qualities) = (Vec::with_capacity(r.triangles), Vec::with_capacity(r.triangles));
    let mut normals = Vec::with_capacity(r.triangles);
    // Each edge (low vertex, high vertex) with the triangles on it.
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let p = tri.map(|i| mesh.positions[i as usize]);
        let (e0, e1, e2) = (sub(p[1], p[0]), sub(p[2], p[1]), sub(p[0], p[2]));
        let n = cross(e0, [-e2[0], -e2[1], -e2[2]]);
        let area2 = len(n);
        let (l0, l1, l2) = (len(e0), len(e1), len(e2));
        let angle = |a: V3, b: V3, la: f32, lb: f32| (dot(a, b) / (la * lb).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees();
        // The angle at each corner, between the two edges that leave it.
        let neg = |a: V3| [-a[0], -a[1], -a[2]];
        let corner = [angle(e0, neg(e2), l0, l2), angle(neg(e0), e1, l0, l1), angle(neg(e1), e2, l1, l2)];
        min_angles.push(corner.iter().cloned().fold(f32::MAX, f32::min));
        // 4 sqrt(3) A with A = area2 / 2.
        qualities.push(if area2 > 0.0 { 2.0 * 3f32.sqrt() * area2 / (l0 * l0 + l1 * l1 + l2 * l2) } else { 0.0 });
        normals.push(if area2 > 0.0 { [n[0] / area2, n[1] / area2, n[2] / area2] } else { [0.0; 3] });
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(t as u32);
        }
    }
    r.min_angle = min_angles.iter().cloned().fold(f32::MAX, f32::min);
    r.thin_fraction = min_angles.iter().filter(|a| **a < 20.0).count() as f32 / r.triangles as f32;
    r.median_min_angle = quantile(&mut min_angles, 0.5);
    r.mean_quality = qualities.iter().sum::<f32>() / r.triangles as f32;
    r.worst_quality = qualities.iter().cloned().fold(f32::MAX, f32::min);

    let mut deviations = Vec::new();
    // Triangles joined across manifold edges form a part; a union-find keeps this linear.
    let mut parent: Vec<u32> = (0..r.triangles as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for tris in edges.values() {
        match tris.len() {
            1 => r.boundary_edges += 1,
            2 => {
                let angle = dot(normals[tris[0] as usize], normals[tris[1] as usize]).clamp(-1.0, 1.0).acos().to_degrees();
                deviations.push(angle);
                let (a, b) = (root(&mut parent, tris[0]), root(&mut parent, tris[1]));
                parent[a as usize] = b;
            }
            _ => {
                r.non_manifold_edges += 1;
                for w in tris.windows(2) {
                    let (a, b) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
                    parent[a as usize] = b;
                }
            }
        }
    }
    r.components = (0..r.triangles as u32).filter(|&i| root(&mut parent, i) == i).count();
    r.max_normal_deviation = deviations.iter().cloned().fold(0.0, f32::max);
    r.p95_normal_deviation = quantile(&mut deviations, 0.95);
    r
}

impl Report {
    /// Adds how far the mesh lies from the zero level of a field. `sample` takes points and returns
    /// (distance, gradient) for each; the distance to the surface is estimated as |d| / |gradient|, at every
    /// vertex and at the centre of every triangle (which is where a coarse mesh deviates most).
    #[allow(dead_code)] // the remesher will call it; tests do for now
    pub fn with_surface_error(mut self, mesh: &RawMesh, sample: &mut dyn FnMut(&[V3]) -> Vec<[f32; 4]>) -> Report {
        let mut points: Vec<V3> = mesh.positions.clone();
        points.extend(mesh.triangles.iter().map(|t| {
            let p = t.map(|i| mesh.positions[i as usize]);
            [0, 1, 2].map(|a| (p[0][a] + p[1][a] + p[2][a]) / 3.0)
        }));
        let errors: Vec<f32> = sample(&points).iter().map(|s| s[0].abs() / len([s[1], s[2], s[3]]).max(1e-6)).collect();
        if !errors.is_empty() {
            self.mean_error = Some(errors.iter().sum::<f32>() / errors.len() as f32);
            self.max_error = Some(errors.iter().cloned().fold(0.0, f32::max));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(positions: Vec<V3>, triangles: Vec<[u32; 3]>) -> RawMesh {
        RawMesh { normals: vec![[0.0; 3]; positions.len()], positions, triangles }
    }

    #[test]
    fn an_equilateral_triangle_is_perfect_and_a_sliver_is_not() {
        let h = 3f32.sqrt() / 2.0;
        let r = report(&mesh(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]], vec![[0, 1, 2]]));
        assert!((r.min_angle - 60.0).abs() < 0.01 && (r.mean_quality - 1.0).abs() < 0.001, "{r}");
        assert_eq!((r.boundary_edges, r.components, r.thin_fraction), (3, 1, 0.0));
        let r = report(&mesh(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.01, 0.0]], vec![[0, 1, 2]]));
        assert!(r.min_angle < 2.0 && r.worst_quality < 0.05 && r.thin_fraction == 1.0, "{r}");
    }

    #[test]
    fn a_tetrahedron_is_watertight_and_two_apart_are_two_parts() {
        let tet = |o: f32| (vec![[o, 0.0, 0.0], [o + 1.0, 0.0, 0.0], [o, 1.0, 0.0], [o, 0.0, 1.0]], vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]);
        let (p, t) = tet(0.0);
        let r = report(&mesh(p.clone(), t.clone()));
        assert!(r.watertight() && r.components == 1, "{r}");
        // Dihedral angles of this corner tetrahedron are 90 degrees at the axes and 54.7 on the slanted face.
        assert!((r.max_normal_deviation - 125.26).abs() < 0.5, "{}", r.max_normal_deviation);
        let (p2, t2) = tet(5.0);
        let mut positions = p;
        positions.extend(p2);
        let mut triangles = t;
        triangles.extend(t2.iter().map(|t| t.map(|i| i + 4)));
        assert_eq!(report(&mesh(positions, triangles)).components, 2);
    }

    #[test]
    fn surface_error_reads_the_field() {
        // A triangle in the plane z = 0.1 against the field z (the plane z = 0): 0.1 away everywhere.
        let m = mesh(vec![[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]], vec![[0, 1, 2]]);
        let r = report(&m).with_surface_error(&m, &mut |pts| pts.iter().map(|p| [p[2], 0.0, 0.0, 1.0]).collect());
        assert!((r.mean_error.unwrap() - 0.1).abs() < 1e-6 && (r.max_error.unwrap() - 0.1).abs() < 1e-6);
    }
}
