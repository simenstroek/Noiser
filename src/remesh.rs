//! Adaptive isotropic remeshing of a closed surface given by a field.
//!
//! Incremental remeshing (Botsch and Kobbelt): split long edges, collapse short ones, flip edges towards
//! the Delaunay criterion, smooth tangentially, and put every vertex back on the field's zero level. The
//! target edge length varies over the surface with its curvature: an edge of length L on a surface of
//! radius of curvature R leaves a chord error of about L²/(8R), so L = sqrt(8 e / curvature) for an
//! error budget e. One number, `e`, sets the triangle count, and it is re-solved against the current mesh
//! every round so the count converges on the target.
//!
//! The field is any function from points to (distance, gradient); see `MeshEvaluator::points`. Vertices
//! are moved onto the surface with Newton steps along the gradient, so the result lies on the true
//! (displaced) surface, not on the coarse mesh it started from.

use crate::mesh::V3;
use crate::surface::RawMesh;

/// Points and a smoothing radius for each in, (distance, gradient x, y, z) out. The radius says over what distance
/// the field may be averaged: a surface with detail finer than the triangles cannot be followed by them, so
/// each vertex is placed on the field blurred at about its own edge length. A radius of 0 asks for the exact field.
pub type Field<'a> = &'a mut dyn FnMut(&[V3], &[f32]) -> Vec<[f32; 4]>;

#[derive(Clone, Debug)]
pub struct Params {
    /// The triangle count to aim for.
    pub target: usize,
    /// Rounds of split, collapse, flip and smooth.
    pub iterations: usize,
    /// Shortest and longest edge the size field may ask for.
    pub min_edge: f32,
    pub max_edge: f32,
    /// The smoothing radius as a fraction of the local edge length. 0 (the default) follows the exact field: blurring
    /// it was tried and made the triangles no cleaner, only the surface less true.
    pub blur: f32,
    /// 0: one edge length everywhere; 1: from a quarter to five times the uniform length, following the curvature.
    pub adaptivity: f32,
}

impl Params {
    /// Sensible edge limits for a mesh with the given bounding box diagonal.
    pub fn new(target: usize, diagonal: f32) -> Params {
        Params { target, iterations: 14, min_edge: diagonal / 1500.0, max_edge: diagonal / 6.0, blur: 0.0, adaptivity: 1.0 }
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
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
fn unit(a: V3) -> V3 {
    let l = len(a);
    if l > 1e-20 { scale(a, 1.0 / l) } else { [0.0; 3] }
}

/// The vertex after `a` then `b` in the triangle, if it has the directed edge a to b.
fn third_of(t: &[u32; 3], a: u32, b: u32) -> Option<u32> {
    (0..3).find(|&i| t[i] == a && t[(i + 1) % 3] == b).map(|i| t[(i + 2) % 3])
}

struct Mesh {
    pos: Vec<V3>,
    nrm: Vec<V3>,
    /// The target edge length at each vertex.
    size: Vec<f32>,
    curv: Vec<f32>,
    /// The live triangles around each vertex; empty means the vertex is gone.
    vt: Vec<Vec<u32>>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    live: usize,
    /// Triangles bucketed by the cells their boxes cover, to find what a moved or new triangle might cross; absent
    /// until `build_grid`. `tri_cells` is the cell box each triangle was entered with.
    grid: Option<Grid>,
    tri_cells: Vec<[i32; 6]>,
}

struct Grid {
    cell: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
}

impl Mesh {
    fn new(raw: RawMesh) -> Mesh {
        let n = raw.positions.len();
        let mut m = Mesh { nrm: vec![[0.0; 3]; n], size: vec![1.0; n], curv: vec![0.0; n], vt: vec![Vec::new(); n], pos: raw.positions, tris: Vec::new(), alive: Vec::new(), live: 0, grid: None, tri_cells: Vec::new() };
        for t in raw.triangles {
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                m.add_tri(t);
            }
        }
        m
    }

    fn add_tri(&mut self, t: [u32; 3]) {
        let id = self.tris.len() as u32;
        self.tris.push(t);
        self.alive.push(true);
        self.live += 1;
        for v in t {
            self.vt[v as usize].push(id);
        }
        self.tri_cells.push([0; 6]);
        self.grid_insert(id);
    }

    fn kill_tri(&mut self, id: u32) {
        self.grid_remove(id);
        self.alive[id as usize] = false;
        self.live -= 1;
        for v in self.tris[id as usize] {
            self.vt[v as usize].retain(|&x| x != id);
        }
    }

    /// Starts keeping the grid, with cells of `cell` units, from the live triangles.
    fn build_grid(&mut self, cell: f32) {
        self.grid = Some(Grid { cell: cell.max(1e-9), cells: std::collections::HashMap::new() });
        for t in 0..self.tris.len() as u32 {
            if self.alive[t as usize] {
                self.grid_insert(t);
            }
        }
    }

    fn cell_box(&self, id: u32, cell: f32) -> [i32; 6] {
        let p = self.tris[id as usize].map(|v| self.pos[v as usize]);
        let mut b = [i32::MAX, i32::MAX, i32::MAX, i32::MIN, i32::MIN, i32::MIN];
        for q in &p {
            for a in 0..3 {
                let c = (q[a] / cell).floor() as i32;
                b[a] = b[a].min(c);
                b[a + 3] = b[a + 3].max(c);
            }
        }
        b
    }

    fn grid_insert(&mut self, id: u32) {
        let Some(cell) = self.grid.as_ref().map(|g| g.cell) else { return };
        let b = self.cell_box(id, cell);
        self.tri_cells[id as usize] = b;
        let g = self.grid.as_mut().unwrap();
        for x in b[0]..=b[3] {
            for y in b[1]..=b[4] {
                for z in b[2]..=b[5] {
                    g.cells.entry((x, y, z)).or_default().push(id);
                }
            }
        }
    }

    fn grid_remove(&mut self, id: u32) {
        let b = self.tri_cells[id as usize];
        let Some(g) = self.grid.as_mut() else { return };
        for x in b[0]..=b[3] {
            for y in b[1]..=b[4] {
                for z in b[2]..=b[5] {
                    if let Some(l) = g.cells.get_mut(&(x, y, z)) {
                        l.retain(|&t| t != id);
                    }
                }
            }
        }
    }

    /// Enters triangle `id` again after its vertices moved.
    fn regrid(&mut self, id: u32) {
        if self.grid.is_some() {
            self.grid_remove(id);
            self.grid_insert(id);
        }
    }

    /// Whether triangle `id` passes through another live triangle (neighbours that only share its vertices do not count).
    fn crosses(&self, id: u32) -> bool {
        let Some(g) = self.grid.as_ref() else { return false };
        let b = self.cell_box(id, g.cell);
        let tri = self.tris[id as usize];
        let pos = |t: [u32; 3]| t.map(|v| self.pos[v as usize]);
        let mine = pos(tri);
        let mut candidates: Vec<u32> = Vec::new();
        for x in b[0]..=b[3] {
            for y in b[1]..=b[4] {
                for z in b[2]..=b[5] {
                    if let Some(l) = g.cells.get(&(x, y, z)) {
                        candidates.extend(l.iter().copied().filter(|&o| o != id));
                    }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for q in &mine {
            for a in 0..3 {
                lo[a] = lo[a].min(q[a]);
                hi[a] = hi[a].max(q[a]);
            }
        }
        for o in candidates {
            if !self.alive[o as usize] {
                continue;
            }
            let other = self.tris[o as usize];
            let theirs = pos(other);
            // Boxes that do not meet cannot cross.
            if (0..3).any(|a| theirs.iter().all(|q| q[a] < lo[a]) || theirs.iter().all(|q| q[a] > hi[a])) {
                continue;
            }
            if crate::meshquality::triangles_cross(mine, tri, theirs, other) {
                return true;
            }
        }
        false
    }

    fn add_vertex(&mut self, p: V3, n: V3, size: f32, curv: f32) -> u32 {
        self.pos.push(p);
        self.nrm.push(n);
        self.size.push(size);
        self.curv.push(curv);
        self.vt.push(Vec::new());
        self.pos.len() as u32 - 1
    }

    /// The live triangles that contain both `a` and `b`.
    fn common(&self, a: u32, b: u32) -> Vec<u32> {
        self.vt[a as usize].iter().copied().filter(|&t| self.tris[t as usize].contains(&b)).collect()
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self.vt[v as usize].iter().flat_map(|&t| self.tris[t as usize]).filter(|&w| w != v).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn edges(&self) -> Vec<(u32, u32)> {
        let mut e = Vec::with_capacity(self.live * 3 / 2 + 8);
        for (t, tri) in self.tris.iter().enumerate() {
            if self.alive[t] {
                for i in 0..3 {
                    let (a, b) = (tri[i], tri[(i + 1) % 3]);
                    e.push((a.min(b), a.max(b)));
                }
            }
        }
        e.sort_unstable();
        e.dedup();
        e
    }

    fn edge_len(&self, a: u32, b: u32) -> f32 {
        len(sub(self.pos[a as usize], self.pos[b as usize]))
    }

    fn target_len(&self, a: u32, b: u32) -> f32 {
        0.5 * (self.size[a as usize] + self.size[b as usize])
    }

    /// Cross product of a triangle's edges (twice its area along its normal), with `moved` standing in for positions.
    fn tri_cross(&self, t: [u32; 3], moved: &[(u32, V3)]) -> V3 {
        let p = |v: u32| moved.iter().find(|(m, _)| *m == v).map_or(self.pos[v as usize], |(_, q)| *q);
        cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0])))
    }

    /// Whether triangle `t` is folded back over a neighbour: across one of its edges the normals are more than
    /// about 125 degrees apart. Operations are refused when they would create such a fold.
    fn fold_at(&self, t: u32) -> bool {
        const FOLD_COS: f32 = -0.57;
        let tri = self.tris[t as usize];
        let n = self.tri_cross(tri, &[]);
        let ln = len(n);
        if ln < 1e-14 {
            return false;
        }
        for i in 0..3 {
            let (x, y) = (tri[i], tri[(i + 1) % 3]);
            let across = self.common(x, y);
            if across.len() != 2 {
                continue;
            }
            let u = if across[0] == t { across[1] } else { across[0] };
            let m = self.tri_cross(self.tris[u as usize], &[]);
            if len(m) > 1e-14 && dot(n, m) < FOLD_COS * ln * len(m) {
                return true;
            }
        }
        false
    }

    // ---- the field ----

    /// Moves every live vertex onto the surface (Newton steps along the gradient), then reads the normals.
    fn project(&mut self, field: Field, steps: usize, blur: f32) {
        let live: Vec<u32> = (0..self.pos.len() as u32).filter(|&v| !self.vt[v as usize].is_empty()).collect();
        for _ in 0..steps {
            let pts: Vec<V3> = live.iter().map(|&v| self.pos[v as usize]).collect();
            let radii: Vec<f32> = live.iter().map(|&v| blur * self.size[v as usize]).collect();
            let mut moves = Vec::with_capacity(live.len());
            for (&v, s) in live.iter().zip(field(&pts, &radii)) {
                let g = [s[1], s[2], s[3]];
                let g2 = dot(g, g);
                if g2 > 1e-12 && s[0].is_finite() {
                    // A vertex never moves more than half its edge length in one step, so it cannot jump to another sheet.
                    let step = scale(g, s[0] / g2);
                    let (l, cap) = (len(step), 0.5 * self.size[v as usize]);
                    let step = if l > cap { scale(step, cap / l) } else { step };
                    moves.push((v, sub(self.pos[v as usize], step)));
                }
            }
            self.commit(moves);
        }
        let pts: Vec<V3> = live.iter().map(|&v| self.pos[v as usize]).collect();
        let radii: Vec<f32> = live.iter().map(|&v| blur * self.size[v as usize]).collect();
        for (&v, s) in live.iter().zip(field(&pts, &radii)) {
            self.nrm[v as usize] = unit([s[1], s[2], s[3]]);
        }
    }

    /// Applies vertex moves, then puts back any vertex whose move folded one of its triangles over.
    fn commit(&mut self, moves: Vec<(u32, V3)>) {
        let old = self.pos.clone();
        // The triangles that were already folded may stay so; no others may become folded.
        let mut around: Vec<u32> = moves.iter().flat_map(|&(v, _)| self.vt[v as usize].iter().copied()).collect();
        around.sort_unstable();
        around.dedup();
        let folded_before: std::collections::HashSet<u32> = around.iter().copied().filter(|&t| self.fold_at(t)).collect();
        for &(v, q) in &moves {
            self.pos[v as usize] = q;
        }
        for &t in &around {
            self.grid_remove(t);
        }
        let was = |t: [u32; 3]| cross(sub(old[t[1] as usize], old[t[0] as usize]), sub(old[t[2] as usize], old[t[0] as usize]));
        for _ in 0..3 {
            let mut bad = Vec::new();
            for &(v, _) in &moves {
                for &t in &self.vt[v as usize] {
                    let tri = self.tris[t as usize];
                    let (before, after) = (was(tri), self.tri_cross(tri, &[]));
                    if dot(before, after) < 0.2 * len(before) * len(after) || (!folded_before.contains(&t) && self.fold_at(t)) {
                        bad.extend(tri);
                    }
                }
            }
            if bad.is_empty() {
                break;
            }
            for v in bad {
                self.pos[v as usize] = old[v as usize];
            }
        }
        if self.grid.is_none() {
            return;
        }
        // No move may push a triangle through another one (vertices that cannot move stay where they were).
        let moved: std::collections::HashSet<u32> = moves.iter().map(|m| m.0).collect();
        for iteration in 0..40 {
            for &t in &around {
                self.grid_insert(t);
            }
            let mut bad = Vec::new();
            for &t in &around {
                if self.crosses(t) {
                    bad.extend(self.tris[t as usize].into_iter().filter(|v| moved.contains(v)));
                }
            }
            if bad.is_empty() {
                if iteration > 5 && std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
                    eprintln!("    commit needed {iteration} guard rounds");
                }
                return;
            }
            for &t in &around {
                self.grid_remove(t);
            }
            for v in bad {
                self.pos[v as usize] = old[v as usize];
            }
        }
        // Out of patience: nothing moves.
        for &(v, _) in &moves {
            self.pos[v as usize] = old[v as usize];
        }
        for &t in &around {
            self.grid_insert(t);
        }
    }

    /// Curvature at each vertex: the largest change of normal along an incident edge, per unit length.
    fn curvature(&mut self) {
        for v in 0..self.pos.len() as u32 {
            if self.vt[v as usize].is_empty() {
                continue;
            }
            let mut k = 0.0f32;
            for w in self.neighbours(v) {
                let d = sub(self.pos[w as usize], self.pos[v as usize]);
                let l2 = dot(d, d);
                if l2 > 1e-24 {
                    k = k.max(dot(sub(self.nrm[w as usize], self.nrm[v as usize]), d).abs() / l2);
                }
            }
            self.curv[v as usize] = k;
        }
    }

    fn raw_size(curv: f32, e: f32, p: &Params) -> f32 {
        (8.0 * e / curv.max(1e-9)).sqrt().clamp(p.min_edge, p.max_edge)
    }

    /// The triangle count the size field for error budget `e` would give on the current mesh.
    fn estimate(&self, e: f32, p: &Params) -> f32 {
        let mut n = 0.0;
        for (t, tri) in self.tris.iter().enumerate() {
            if self.alive[t] {
                let a = 0.5 * len(self.tri_cross(*tri, &[]));
                let l = tri.iter().map(|&v| Self::raw_size(self.curv[v as usize], e, p)).sum::<f32>() / 3.0;
                n += a / (0.433 * l * l);
            }
        }
        n
    }

    /// The error budget that gives the target count, by bisection (the count falls as `e` grows).
    fn solve_e(&self, p: &Params, target: f32) -> f32 {
        let (mut lo, mut hi) = (1e-9f32.ln(), 10f32.ln());
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if self.estimate(mid.exp(), p) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (0.5 * (lo + hi)).exp()
    }

    /// Edge-length targets from the curvature, kept from changing faster than the edges are long.
    fn set_sizes(&mut self, e: f32, p: &Params) {
        for v in 0..self.pos.len() {
            self.size[v] = Self::raw_size(self.curv[v], e, p);
        }
        let edges = self.edges();
        for _ in 0..3 {
            for &(a, b) in &edges {
                let d = 0.7 * self.edge_len(a, b);
                let (sa, sb) = (self.size[a as usize], self.size[b as usize]);
                self.size[a as usize] = sa.min(sb + d);
                self.size[b as usize] = sb.min(sa + d);
            }
        }
    }

    // ---- operations ----

    fn split(&mut self, a: u32, b: u32) -> bool {
        let ts = self.common(a, b);
        if ts.len() != 2 {
            return false;
        }
        let (mut one, mut two) = (None, None);
        for &t in &ts {
            if let Some(c) = third_of(&self.tris[t as usize], a, b) {
                one = Some((t, c));
            } else if let Some(d) = third_of(&self.tris[t as usize], b, a) {
                two = Some((t, d));
            }
        }
        let (Some((t1, c)), Some((t2, d))) = (one, two) else { return false };
        let (ia, ib) = (a as usize, b as usize);
        let m = self.add_vertex(scale(add(self.pos[ia], self.pos[ib]), 0.5), unit(add(self.nrm[ia], self.nrm[ib])), 0.5 * (self.size[ia] + self.size[ib]), self.curv[ia].max(self.curv[ib]));
        self.kill_tri(t1);
        self.kill_tri(t2);
        self.add_tri([a, m, c]);
        self.add_tri([m, b, c]);
        self.add_tri([b, m, d]);
        self.add_tri([m, a, d]);
        true
    }

    /// Merges `a` into `b` (at the edge's midpoint) unless that would fold or pinch the surface.
    fn collapse(&mut self, a: u32, b: u32, longest: f32) -> bool {
        let ts = self.common(a, b);
        if ts.len() != 2 {
            return false;
        }
        let others: Vec<u32> = ts.iter().map(|&t| *self.tris[t as usize].iter().find(|&&v| v != a && v != b).unwrap()).collect();
        if others[0] == others[1] {
            return false;
        }
        // The link condition: a and b may share only the two vertices opposite their edge.
        let (na, nb) = (self.neighbours(a), self.neighbours(b));
        let shared = na.iter().filter(|v| nb.contains(v)).count();
        if shared != 2 {
            return false;
        }
        let m = scale(add(self.pos[a as usize], self.pos[b as usize]), 0.5);
        let moved = [(a, m), (b, m)];
        for &v in &[a, b] {
            for &t in &self.vt[v as usize] {
                if ts.contains(&t) {
                    continue;
                }
                let tri = self.tris[t as usize];
                let (old, new) = (self.tri_cross(tri, &[]), self.tri_cross(tri, &moved));
                if len(new) < 1e-12 || dot(old, new) < 0.3 * len(old) * len(new) {
                    return false;
                }
            }
        }
        if na.iter().chain(&nb).any(|&x| x != a && x != b && len(sub(self.pos[x as usize], m)) > longest) {
            return false;
        }
        let was_folded = self.vt[a as usize].iter().chain(&self.vt[b as usize]).any(|&t| self.fold_at(t));
        let saved: Vec<[u32; 3]> = self.vt[a as usize].iter().map(|&t| self.tris[t as usize]).collect();
        let (old_pos, old_nrm, old_curv) = (self.pos[b as usize], self.nrm[b as usize], self.curv[b as usize]);
        let first = self.tris.len() as u32;
        for &t in &ts {
            self.kill_tri(t);
        }
        for t in self.vt[a as usize].clone() {
            let mut tri = self.tris[t as usize];
            self.kill_tri(t);
            for v in &mut tri {
                if *v == a {
                    *v = b;
                }
            }
            self.add_tri(tri);
        }
        self.pos[b as usize] = m;
        self.nrm[b as usize] = unit(add(self.nrm[a as usize], self.nrm[b as usize]));
        self.curv[b as usize] = self.curv[a as usize].max(self.curv[b as usize]);
        for t in self.vt[b as usize].clone() {
            self.regrid(t);
        }
        let folds = !was_folded && self.vt[b as usize].clone().iter().any(|&t| self.fold_at(t));
        if folds || self.vt[b as usize].clone().iter().any(|&t| self.crosses(t)) {
            // Put everything back as it was.
            for t in first..self.tris.len() as u32 {
                if self.alive[t as usize] {
                    self.kill_tri(t);
                }
            }
            (self.pos[b as usize], self.nrm[b as usize], self.curv[b as usize]) = (old_pos, old_nrm, old_curv);
            for tri in saved {
                self.add_tri(tri);
            }
            return false;
        }
        true
    }

    /// Flips the edge when the two opposite angles add up to more than a straight angle (not Delaunay).
    fn flip(&mut self, a: u32, b: u32) -> bool {
        let ts = self.common(a, b);
        if ts.len() != 2 {
            return false;
        }
        let (mut one, mut two) = (None, None);
        for &t in &ts {
            if let Some(c) = third_of(&self.tris[t as usize], a, b) {
                one = Some((t, c));
            } else if let Some(d) = third_of(&self.tris[t as usize], b, a) {
                two = Some((t, d));
            }
        }
        let (Some((t1, c)), Some((t2, d))) = (one, two) else { return false };
        if c == d || !self.common(c, d).is_empty() {
            return false;
        }
        let p = |v: u32| self.pos[v as usize];
        let angle = |at: u32| {
            let (u, w) = (unit(sub(p(a), p(at))), unit(sub(p(b), p(at))));
            dot(u, w).clamp(-1.0, 1.0).acos()
        };
        if angle(c) + angle(d) <= std::f32::consts::PI + 1e-3 {
            return false;
        }
        // The new triangles must still face the way the old pair did.
        let avg = unit(add(unit(self.tri_cross(self.tris[t1 as usize], &[])), unit(self.tri_cross(self.tris[t2 as usize], &[]))));
        for tri in [[a, d, c], [d, b, c]] {
            let n = self.tri_cross(tri, &[]);
            if len(n) < 1e-12 || dot(n, avg) < 0.5 * len(n) {
                return false;
            }
        }
        let was_folded = self.fold_at(t1) || self.fold_at(t2);
        let (old1, old2) = (self.tris[t1 as usize], self.tris[t2 as usize]);
        self.kill_tri(t1);
        self.kill_tri(t2);
        let first = self.tris.len() as u32;
        self.add_tri([a, d, c]);
        self.add_tri([d, b, c]);
        if (!was_folded && (self.fold_at(first) || self.fold_at(first + 1))) || self.crosses(first) || self.crosses(first + 1) {
            self.kill_tri(first);
            self.kill_tri(first + 1);
            self.add_tri(old1);
            self.add_tri(old2);
            return false;
        }
        true
    }

    fn split_pass(&mut self) -> usize {
        let mut n = 0;
        for (a, b) in self.edges() {
            if self.edge_len(a, b) > 4.0 / 3.0 * self.target_len(a, b) && self.split(a, b) {
                n += 1;
            }
        }
        n
    }

    fn collapse_pass(&mut self) -> usize {
        let mut n = 0;
        for (a, b) in self.edges() {
            if self.vt[a as usize].is_empty() || self.vt[b as usize].is_empty() {
                continue;
            }
            let target = self.target_len(a, b);
            if self.edge_len(a, b) < 0.8 * target && self.collapse(a, b, 4.0 / 3.0 * target) {
                n += 1;
            }
        }
        n
    }

    /// Removes slivers: for each triangle of poor shape (4 sqrt(3) A / (a² + b² + c²) under `min_quality`) the shortest
    /// edge is collapsed, or when that is not allowed the longest edge is flipped.
    fn sliver_pass(&mut self, min_quality: f32) -> usize {
        let mut n = 0;
        for t in 0..self.tris.len() as u32 {
            if !self.alive[t as usize] {
                continue;
            }
            let tri = self.tris[t as usize];
            let l: [f32; 3] = std::array::from_fn(|i| self.edge_len(tri[i], tri[(i + 1) % 3]));
            let area2 = len(self.tri_cross(tri, &[]));
            let q = 2.0 * 3f32.sqrt() * area2 / (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).max(1e-30);
            if q >= min_quality {
                continue;
            }
            let short = (0..3).min_by(|&i, &j| l[i].total_cmp(&l[j])).unwrap();
            let long = (0..3).max_by(|&i, &j| l[i].total_cmp(&l[j])).unwrap();
            let (a, b) = (tri[short], tri[(short + 1) % 3]);
            // Either end may be the one to go; try both, since one of them can be blocked by the link condition.
            let reach = 3.0 * self.target_len(a, b).max(l[long]);
            if self.collapse(a, b, reach) || self.collapse(b, a, reach) || self.flip(tri[long], tri[(long + 1) % 3]) {
                n += 1;
            }
        }
        n
    }

    fn flip_pass(&mut self) -> usize {
        let mut n = 0;
        for (a, b) in self.edges() {
            if self.flip(a, b) {
                n += 1;
            }
        }
        n
    }

    /// Moves each vertex towards the centre of its neighbours, along the surface only.
    fn smooth_pass(&mut self) {
        let mut moves: Vec<(u32, V3)> = Vec::new();
        for v in 0..self.pos.len() as u32 {
            let ring = self.neighbours(v);
            // Only vertices on a closed fan of manifold edges move.
            if ring.len() < 3 || self.vt[v as usize].len() != ring.len() || ring.iter().any(|&w| self.common(v, w).len() != 2) {
                continue;
            }
            let c = scale(ring.iter().fold([0.0; 3], |s, &w| add(s, self.pos[w as usize])), 1.0 / ring.len() as f32);
            let d = sub(c, self.pos[v as usize]);
            let n = self.nrm[v as usize];
            let tangent = sub(d, scale(n, dot(d, n)));
            moves.push((v, add(self.pos[v as usize], scale(tangent, 0.5))));
        }
        self.commit(moves);
    }

    /// Debug aid: how many live triangles cross another.
    fn crossing_now(&self) -> usize {
        let snap = Mesh { pos: self.pos.clone(), nrm: self.nrm.clone(), size: self.size.clone(), curv: self.curv.clone(), vt: self.vt.clone(), tris: self.tris.clone(), alive: self.alive.clone(), live: self.live, grid: None, tri_cells: Vec::new() };
        crate::meshquality::self_intersecting_triangles(&snap.finish()).iter().filter(|b| **b).count()
    }

    fn finish(self) -> RawMesh {
        let mut map = vec![u32::MAX; self.pos.len()];
        let mut out = RawMesh::default();
        for (v, p) in self.pos.iter().enumerate() {
            if !self.vt[v].is_empty() {
                map[v] = out.positions.len() as u32;
                out.positions.push(*p);
                out.normals.push(self.nrm[v]);
            }
        }
        for (t, tri) in self.tris.iter().enumerate() {
            if self.alive[t] {
                out.triangles.push(tri.map(|v| map[v as usize]));
            }
        }
        out
    }
}

/// Remeshes the closed surface `start` (counter-clockwise seen from outside) onto the zero level of `field`.
/// `progress` is called with (round, rounds).
pub fn remesh(start: RawMesh, field: Field, params: &Params, progress: &dyn Fn(usize, usize)) -> RawMesh {
    let mut m = Mesh::new(start);
    if m.live == 0 {
        return RawMesh::default();
    }
    // Edges stay within a few times the length that would give the target count on a uniform mesh: a sharp
    // crease has a huge curvature but must not eat the whole triangle budget.
    let area: f32 = m.tris.iter().enumerate().filter(|(t, _)| m.alive[*t]).map(|(_, t)| 0.5 * len(m.tri_cross(*t, &[]))).sum();
    let uniform = (4.0 * area / (3f32.sqrt() * params.target as f32)).sqrt();
    let a = params.adaptivity.clamp(0.0, 1.0);
    let p = &Params { min_edge: params.min_edge.max(uniform * (1.0 - 0.75 * a)), max_edge: params.max_edge.min(uniform * (1.0 + 4.0 * a)).max(uniform * (1.0 - 0.5 * a)), ..params.clone() };
    m.size.iter_mut().for_each(|s| *s = 0.25 * p.max_edge);
    m.build_grid(uniform.max(1e-9));
    if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
        let raw = Mesh { pos: m.pos.clone(), nrm: m.nrm.clone(), size: m.size.clone(), curv: m.curv.clone(), vt: m.vt.clone(), tris: m.tris.clone(), alive: m.alive.clone(), live: m.live, grid: None, tri_cells: Vec::new() }.finish();
        eprintln!("  crossing at the start: {}", crate::meshquality::self_intersecting_triangles(&raw).iter().filter(|b| **b).count());
    }
    m.project(field, 2, p.blur);
    let rounds = p.iterations;
    let mut extra = 0;
    let mut round = 0;
    // How many more triangles the edge-length rules end up with than the equilateral estimate says.
    let mut corr = 1.0f32;
    while round < rounds + extra {
        progress(round, rounds + extra);
        m.curvature();
        let e = m.solve_e(p, p.target as f32 / corr);
        m.set_sizes(e, p);
        let before = m.live;
        let dbg = std::env::var_os("NOISER_REMESH_DEBUG").is_some() && round < 2;
        let c0 = if dbg { m.crossing_now() } else { 0 };
        let sp = m.split_pass();
        let c1 = if dbg { m.crossing_now() } else { 0 };
        let co = m.collapse_pass();
        let c2 = if dbg { m.crossing_now() } else { 0 };
        let fl = m.flip_pass();
        let c3 = if dbg { m.crossing_now() } else { 0 };
        m.smooth_pass();
        let c4 = if dbg { m.crossing_now() } else { 0 };
        let fl2 = m.flip_pass();
        if dbg {
            eprintln!("  crossings: start {c0}, split {c1}, collapse {c2}, flip {c3}, smooth {c4}, flip {}", m.crossing_now());
        }
        let ratio = m.live as f32 / m.estimate(e, p).max(1.0);
        corr = if round == 0 { ratio } else { 0.5 * corr + 0.5 * ratio };
        if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
            eprintln!("round {round}: e {e:.2e}, {before} -> {} triangles ({sp} splits, {co} collapses, {fl}+{fl2} flips), estimated {:.0}", m.live, m.estimate(e, p));
        }
        m.project(field, 1, p.blur);
        if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
            let snap = Mesh { grid: None, ..Mesh { pos: m.pos.clone(), nrm: m.nrm.clone(), size: m.size.clone(), curv: m.curv.clone(), vt: m.vt.clone(), tris: m.tris.clone(), alive: m.alive.clone(), live: m.live, grid: None, tri_cells: Vec::new() } };
            let raw = snap.finish();
            eprintln!("  crossing after round {round}: {}", crate::meshquality::self_intersecting_triangles(&raw).iter().filter(|b| **b).count());
        }
        round += 1;
        // The count settles late in the run; a few more rounds if it is still off by more than 6%.
        if round == rounds + extra && extra < 6 && (m.live as f32 - p.target as f32).abs() > 0.06 * p.target as f32 {
            extra += 2;
        }
    }
    // Clean up slivers, then settle.
    for _ in 0..3 {
        if m.sliver_pass(0.25) == 0 {
            break;
        }
        m.flip_pass();
        m.smooth_pass();
        m.project(field, 1, p.blur);
    }
    m.project(field, 2, p.blur);
    progress(rounds + extra, rounds + extra);
    m.finish()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshquality;

    fn icosphere(radius: f32, subdivisions: u32) -> RawMesh {
        let m = crate::mesh::shapes::icosphere(radius, subdivisions);
        RawMesh { normals: vec![[0.0; 3]; m.positions.len()], positions: m.positions, triangles: m.triangles }
    }

    fn sphere_field(pts: &[V3], _radii: &[f32]) -> Vec<[f32; 4]> {
        pts.iter()
            .map(|p| {
                let r = len(*p).max(1e-9);
                [r - 1.0, p[0] / r, p[1] / r, p[2] / r]
            })
            .collect()
    }

    #[test]
    fn a_sphere_is_remeshed_to_the_requested_count_with_good_triangles() {
        for target in [500usize, 4000, 20000] {
            let out = remesh(icosphere(1.0, 3), &mut sphere_field, &Params::new(target, 3.46), &|_, _| {});
            let r = meshquality::report(&out).with_surface_error(&out, &mut |pts| sphere_field(pts, &[]));
            eprintln!("target {target}:\n{r}");
            assert!((r.triangles as f32 - target as f32).abs() < 0.15 * target as f32, "{} triangles for a target of {target}", r.triangles);
            assert!(r.watertight() && r.components == 1, "not a single closed surface: {r}");
            assert!(r.thin_fraction < 0.03 && r.min_angle > 15.0, "thin triangles: {r}");
            // Vertices sit on the surface; the error left is the chord sag at triangle centres, which falls as 1 / count.
            let worst_vertex = out.positions.iter().map(|p| (len(*p) - 1.0).abs()).fold(0.0, f32::max);
            assert!(worst_vertex < 1e-4, "a vertex is {worst_vertex} off the sphere");
            assert!(r.max_error.unwrap() < 10.0 / target as f32, "chord error {:?}", r.max_error);
        }
    }

    #[test]
    fn edges_are_shorter_where_the_surface_bends_more() {
        // An ellipsoid with semi-axes 1.6, 0.5 and 0.5: its tips curve much more sharply than its flanks.
        let field = |pts: &[V3], _radii: &[f32]| -> Vec<[f32; 4]> {
            pts.iter()
                .map(|p| {
                    let q = [p[0] / 1.6, p[1] / 0.5, p[2] / 0.5];
                    let k = len(q).max(1e-9);
                    // The standard first-order distance estimate to an ellipsoid.
                    let g = [q[0] / 1.6 / k, q[1] / 0.5 / k, q[2] / 0.5 / k];
                    let gl = len(g).max(1e-9);
                    [(k - 1.0) / gl, g[0] / gl, g[1] / gl, g[2] / gl]
                })
                .collect()
        };
        let mut start = icosphere(1.0, 3);
        for p in &mut start.positions {
            *p = [p[0] * 1.6, p[1] * 0.5, p[2] * 0.5];
        }
        let out = remesh(start, &mut { field }, &Params::new(3000, 3.5), &|_, _| {});
        let mean_edge = |keep: &dyn Fn(V3) -> bool| {
            let (mut sum, mut n) = (0.0, 0);
            for t in &out.triangles {
                for i in 0..3 {
                    let (a, b) = (out.positions[t[i] as usize], out.positions[t[(i + 1) % 3] as usize]);
                    if keep(a) && keep(b) {
                        sum += len(sub(a, b));
                        n += 1;
                    }
                }
            }
            sum / n.max(1) as f32
        };
        let (tip, flank) = (mean_edge(&|p| p[0].abs() > 1.3), mean_edge(&|p| p[0].abs() < 0.4));
        eprintln!("{} triangles, edge at the tips {tip:.4}, on the flanks {flank:.4}", out.triangles.len());
        assert!(tip < 0.7 * flank, "tips {tip}, flanks {flank}");
        // With no adaptivity the same surface gets even triangles.
        let mut start = icosphere(1.0, 3);
        for p in &mut start.positions {
            *p = [p[0] * 1.6, p[1] * 0.5, p[2] * 0.5];
        }
        let even = remesh(start, &mut { field }, &Params { adaptivity: 0.0, ..Params::new(3000, 3.5) }, &|_, _| {});
        let mean_even = |keep: &dyn Fn(V3) -> bool| {
            let (mut sum, mut n) = (0.0, 0);
            for t in &even.triangles {
                for i in 0..3 {
                    let (a, b) = (even.positions[t[i] as usize], even.positions[t[(i + 1) % 3] as usize]);
                    if keep(a) && keep(b) {
                        sum += len(sub(a, b));
                        n += 1;
                    }
                }
            }
            sum / n.max(1) as f32
        };
        let (tip_e, flank_e) = (mean_even(&|p| p[0].abs() > 1.3), mean_even(&|p| p[0].abs() < 0.4));
        assert!(tip_e > 0.85 * flank_e, "even: tips {tip_e}, flanks {flank_e}");
        let r = meshquality::report(&out);
        assert!(r.watertight() && r.thin_fraction < 0.05, "{r}");
    }

    #[test]
    fn a_bumpy_sphere_stays_a_clean_surface_at_every_count() {
        // Radius 1 plus bumps of wavelength about 0.7 and height 0.08: finer than the triangles at low counts.
        let bumpy = |pts: &[V3], _radii: &[f32]| -> Vec<[f32; 4]> {
            let f = |p: V3| len(p) - 1.0 - 0.08 * (9.0 * p[0]).sin() * (9.0 * p[1]).sin() * (9.0 * p[2]).sin();
            let e = 1e-3;
            pts.iter()
                .map(|p| {
                    let g = [0, 1, 2].map(|a| {
                        let (mut hi, mut lo) = (*p, *p);
                        hi[a] += e;
                        lo[a] -= e;
                        (f(hi) - f(lo)) / (2.0 * e)
                    });
                    [f(*p), g[0], g[1], g[2]]
                })
                .collect()
        };
        for target in [800usize, 5000, 30000] {
            let out = remesh(icosphere(1.0, 4), &mut { bumpy }, &Params::new(target, 3.5), &|_, _| {});
            let r = meshquality::report(&out);
            eprintln!("target {target}:
{r}");
            assert!(r.watertight() && r.components == 1, "{r}");
            assert!(r.max_normal_deviation < 120.0 && r.thin_fraction < 0.05, "{r}");
            let crossing = meshquality::self_intersecting_triangles(&out).iter().filter(|b| **b).count();
            assert_eq!(crossing, 0, "{crossing} triangles cross another at {target} triangles");
        }
    }

    #[test]
    #[ignore = "experiment: a surface rougher than the triangles"]
    fn rough_sphere_experiment() {
        let rough = |pts: &[V3], _radii: &[f32]| -> Vec<[f32; 4]> {
            let f = |p: V3| {
                let b = |k: f32| (k * p[0]).sin() * (k * p[1] + 1.0).sin() * (k * p[2] + 2.0).sin();
                len(p) - 1.0 - 0.14 * b(12.0) - 0.07 * b(24.0) - 0.035 * b(48.0)
            };
            let e = 1e-3;
            pts.iter()
                .map(|p| {
                    let g = [0, 1, 2].map(|a| {
                        let (mut hi, mut lo) = (*p, *p);
                        hi[a] += e;
                        lo[a] -= e;
                        (f(hi) - f(lo)) / (2.0 * e)
                    });
                    [f(*p), g[0], g[1], g[2]]
                })
                .collect()
        };
        for target in [800usize, 5000, 30000] {
            let out = remesh(icosphere(1.0, 4), &mut { rough }, &Params::new(target, 3.5), &|_, _| {});
            eprintln!("target {target}:
{}", meshquality::report(&out));
        }
    }
}
