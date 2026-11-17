// The displacement mask (see meshmask.rs). It is compiled only when a mesh with mask terms is in use:
// its many inlined distance-field lookups added about 0.65 s to every pipeline build. Needs noise.wgsl
// (mesh_sdf_at, safe_normalize).

// Displacement strength at `p` (mesh space) from the mask terms (meshmask.rs); 1 with no terms.
// `terms` holds two vec4 per term: (kind, weight, lo, hi) and (direction xyz, invert). Every term is
// measured on the mesh's own distance field, at the nearest surface point, so the mask is a property
// of the surface and does not change along the normal. `blo`/`bhi` are the mesh's bounds.
fn mesh_mask(t: texture_3d<f32>, p: vec3f, origin: vec3f, voxel: f32, blo: vec3f, bhi: vec3f, terms: array<vec4f, 12>, count: u32) -> f32 {
    if (count == 0u) { return 1.0; }
    let e = 1.5 * voxel;
    let d0 = mesh_sdf_at(t, p, origin, voxel);
    let xp = mesh_sdf_at(t, p + vec3f(e, 0.0, 0.0), origin, voxel);
    let xm = mesh_sdf_at(t, p - vec3f(e, 0.0, 0.0), origin, voxel);
    let yp = mesh_sdf_at(t, p + vec3f(0.0, e, 0.0), origin, voxel);
    let ym = mesh_sdf_at(t, p - vec3f(0.0, e, 0.0), origin, voxel);
    let zp = mesh_sdf_at(t, p + vec3f(0.0, 0.0, e), origin, voxel);
    let zm = mesh_sdf_at(t, p - vec3f(0.0, 0.0, e), origin, voxel);
    let n = safe_normalize(vec3f(xp - xm, yp - ym, zp - zm));
    let s = p - n * d0; // the nearest surface point (one Newton step on a distance field)
    var sum = 0.0;
    for (var i = 0u; i < count; i++) {
        let a = terms[2u * i];
        let b = terms[2u * i + 1u];
        let kind = u32(a.x);
        if (kind == 1u) { sum += a.y; continue; }
        var raw = 0.0;
        if (kind == 2u) {
            let u01 = (s - blo) / max(bhi - blo, vec3f(1e-6));
            raw = dot(select(u01, vec3f(1.0) - u01, b.xyz < vec3f(0.0)), abs(b.xyz));
        } else if (kind == 3u) {
            raw = dot(n, b.xyz);
        } else if (kind == 4u) {
            // Ambient occlusion from the distance field (IQ): how far the field falls short of the
            // distance moved along the normal, summed over a few steps.
            var occ = 0.0;
            var sca = 1.0;
            for (var k = 1; k <= 5; k++) {
                let h = 0.01 + 0.07 * f32(k);
                occ += (h - mesh_sdf_at(t, s + n * h, origin, voxel)) * sca;
                sca *= 0.85;
            }
            raw = clamp(1.0 - 2.0 * occ, 0.0, 1.0);
        } else if (kind == 5u) {
            // March inward along the normal until the field turns positive again.
            var q = s - n * 0.01;
            var depth = 0.01;
            for (var k = 0; k < 24; k++) {
                let dd = mesh_sdf_at(t, q, origin, voxel);
                if (dd >= 0.0) { break; }
                let step = max(-dd, 0.5 * voxel);
                depth += step;
                q -= n * step;
            }
            raw = depth;
        } else {
            raw = 0.5 * (xp + xm + yp + ym + zp + zm - 6.0 * d0) / (e * e); // mean curvature
        }
        var v = clamp((raw - a.z) / max(a.w - a.z, 1e-6), 0.0, 1.0);
        if (b.w > 0.5) { v = 1.0 - v; }
        sum += a.y * v;
    }
    return sum;
}
