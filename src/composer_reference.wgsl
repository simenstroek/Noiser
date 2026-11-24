// The reference model of the SDF Composer: an ordinary rasterised mesh, drawn into a colour target and a target that
// holds each pixel's distance from the eye, which composer_view.wgsl reads to put the composition in front of or
// behind it.

struct R {
    eye: vec4f,
    forward: vec4f,
    right: vec4f,
    up: vec4f,
    light: vec4f,
    // aspect, tan(fov / 2), near, far
    params: vec4f,
    // the model matrix: three rows, the last column is the translation
    row0: vec4f,
    row1: vec4f,
    row2: vec4f,
}

@group(0) @binding(0) var<uniform> r: R;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
}

@vertex
fn vs(in: VsIn) -> VsOut {
    let p = vec4f(in.position, 1.0);
    let world = vec3f(dot(r.row0, p), dot(r.row1, p), dot(r.row2, p));
    let normal = vec3f(dot(r.row0.xyz, in.normal), dot(r.row1.xyz, in.normal), dot(r.row2.xyz, in.normal));
    let d = world - r.eye.xyz;
    let vz = dot(d, r.forward.xyz);
    let vx = dot(d, r.right.xyz);
    let vy = dot(d, r.up.xyz);
    let near = r.params.z;
    let far = r.params.w;
    var out: VsOut;
    out.clip = vec4f(vx / (r.params.y * r.params.x), vy / r.params.y, far * (vz - near) / (far - near), vz);
    out.world = world;
    out.normal = normal;
    return out;
}

struct FsOut {
    @location(0) color: vec4f,
    @location(1) distance: vec4f,
}

@fragment
fn fs(in: VsOut) -> FsOut {
    let to_eye = r.eye.xyz - in.world;
    let view = normalize(to_eye);
    var n = normalize(in.normal);
    // Two-sided: a mesh with mixed winding still lights from the side that faces the eye.
    if (dot(n, view) < 0.0) { n = -n; }
    let diffuse = max(dot(n, r.light.xyz), 0.0);
    let rim = pow(1.0 - max(dot(n, view), 0.0), 3.0) * 0.15;
    let shade = 0.28 + 0.72 * diffuse + rim;
    var out: FsOut;
    out.color = vec4f(min(vec3f(0.55, 0.64, 0.78) * shade, vec3f(1.0)), 1.0);
    out.distance = vec4f(length(to_eye), 0.0, 0.0, 0.0);
    return out;
}
