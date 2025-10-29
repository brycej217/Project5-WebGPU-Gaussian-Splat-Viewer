struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) col_op: vec2u,
    @location(1) conic: vec3<f32>,
    @location(2) pixelCenter: vec2<f32>
};

struct Splat {
    pos: u32,
    size: u32,
    col_op: vec2u,
    conic: vec2u
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(@builtin(instance_index) iIdx : u32, @builtin(vertex_index) vIdx: u32) -> VertexOutput 
{
    let splat = splats[sort_indices[iIdx]];
    let center = unpack2x16float(splat.pos);
    let size = unpack2x16float(splat.size);
    let conic = vec3f(unpack2x16float(splat.conic[0]), unpack2x16float(splat.conic[1]).x);

    // quad vertices
    const vertices = array<vec2f, 4>(
        vec2f(-1, -1),
        vec2f( 1, -1),
        vec2f(-1,  1),
        vec2f( 1,  1)
    );

    var out: VertexOutput;
    out.position = vec4f(center + (size * vertices[vIdx]), 0.0, 1.0);
    out.col_op = splat.col_op;
    out.conic = conic;
    out.pixelCenter = (center.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5)) * camera.viewport;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let col_op= vec4f(
        unpack2x16float(in.col_op.x),
        unpack2x16float(in.col_op.y)
    );

    let dist = in.pixelCenter - in.position.xy;
    let power = -0.5 * (in.conic.x * pow(dist.x, 2.0) 
                      + in.conic.z * pow(dist.y, 2.0)) 
                      - in.conic.y * dist.x * dist.y;

    if (power > 0.0) 
    {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let alpha = min(0.99f, col_op.w * exp(power));

    return vec4f(col_op.rgb, 1.f) * alpha;
}