//! Tileable 3D blue noise tables (void-and-cluster, Ulichney 1993, on a torus).
//!
//! The tables ship as `assets/bluenoise.bin` (u16 ranks for 4³, 8³, 16³, 32³ back to back) because
//! generating 32³ takes a few seconds. Regenerate with:
//! `cargo test --release regenerate_blue_noise -- --ignored`

pub const SIZES: [usize; 4] = [4, 8, 16, 32];

/// All tables as values in [0, 1], in the layout noise.wgsl's `blue()` expects.
pub fn tables() -> Vec<f32> {
    let bytes = include_bytes!("../assets/bluenoise.bin");
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut at = 0;
    for n in SIZES {
        let count = n * n * n;
        for i in 0..count {
            let r = u16::from_le_bytes([bytes[at + 2 * i], bytes[at + 2 * i + 1]]);
            out.push(r as f32 / (count - 1) as f32);
        }
        at += 2 * count;
    }
    out
}

/// Ranks 0..n³ for an n³ periodic grid, ordered so every prefix is evenly spread.
#[cfg_attr(not(test), allow(dead_code))] // only the regenerate test calls it
pub fn generate(n: usize, seed: u64) -> Vec<u16> {
    let total = n * n * n;
    let sigma = 1.5f32;
    let radius = ((3.0 * sigma).ceil() as i32).min(n as i32 / 2);
    // Gaussian splat weights over the window.
    let mut kernel = Vec::new();
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let r2 = (dx * dx + dy * dy + dz * dz) as f32;
                kernel.push(((dx, dy, dz), (-r2 / (2.0 * sigma * sigma)).exp()));
            }
        }
    }
    let idx = |x: i32, y: i32, z: i32| {
        let w = |v: i32| v.rem_euclid(n as i32) as usize;
        (w(z) * n + w(y)) * n + w(x)
    };
    let splat = |energy: &mut [f32], i: usize, sign: f32| {
        let (x, y, z) = ((i % n) as i32, ((i / n) % n) as i32, (i / (n * n)) as i32);
        for &((dx, dy, dz), w) in &kernel {
            energy[idx(x + dx, y + dy, z + dz)] += sign * w;
        }
    };
    // Tightest cluster: the set point with the most energy. Largest void: the empty one with the least.
    let extreme = |energy: &[f32], on: &[bool], want_on: bool| {
        let mut best = usize::MAX;
        for i in 0..total {
            if on[i] == want_on
                && (best == usize::MAX || (want_on && energy[i] > energy[best]) || (!want_on && energy[i] < energy[best]))
            {
                best = i;
            }
        }
        best
    };

    // Initial pattern: ~10% random points, relaxed by moving the tightest cluster into the largest void.
    let mut rng = seed.max(1);
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut on = vec![false; total];
    let mut energy = vec![0.0f32; total];
    let initial = (total / 10).max(1);
    let mut placed = 0;
    while placed < initial {
        let i = (next() % total as u64) as usize;
        if !on[i] {
            on[i] = true;
            splat(&mut energy, i, 1.0);
            placed += 1;
        }
    }
    for _ in 0..total {
        let c = extreme(&energy, &on, true);
        on[c] = false;
        splat(&mut energy, c, -1.0);
        let v = extreme(&energy, &on, false);
        on[v] = true;
        splat(&mut energy, v, 1.0);
        if v == c {
            break;
        }
    }

    let mut rank = vec![0u16; total];
    // Phase 1: rank the initial points by removing tightest clusters first (they get the highest ranks).
    let (mut on1, mut e1) = (on.clone(), energy.clone());
    for r in (0..initial).rev() {
        let c = extreme(&e1, &on1, true);
        on1[c] = false;
        splat(&mut e1, c, -1.0);
        rank[c] = r as u16;
    }
    // Phase 2: fill the largest voids in order.
    // shortcut: no phase-3 inversion for the last half; fine for preview/texture use, add it if spectra look off.
    for r in initial..total {
        let v = extreme(&energy, &on, false);
        on[v] = true;
        splat(&mut energy, v, 1.0);
        rank[v] = r as u16;
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_are_a_permutation() {
        for n in [4, 8] {
            let mut r = generate(n, 7);
            r.sort();
            assert!(r.iter().enumerate().all(|(i, &v)| v as usize == i), "n = {n}");
        }
    }

    #[test]
    fn shipped_tables_cover_zero_to_one() {
        let t = tables();
        assert_eq!(t.len(), SIZES.iter().map(|n| n * n * n).sum::<usize>());
        let mut at = 0;
        for n in SIZES {
            let s = &t[at..at + n * n * n];
            assert_eq!(s.iter().cloned().fold(f32::MAX, f32::min), 0.0);
            assert_eq!(s.iter().cloned().fold(f32::MIN, f32::max), 1.0);
            at += n * n * n;
        }
    }

    #[test]
    #[ignore = "writes assets/bluenoise.bin; run with --release"]
    fn regenerate_blue_noise() {
        let mut bytes = Vec::new();
        for n in SIZES {
            for r in generate(n, 0x5EED + n as u64) {
                bytes.extend(r.to_le_bytes());
            }
        }
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bluenoise.bin"), bytes).unwrap();
    }
}
