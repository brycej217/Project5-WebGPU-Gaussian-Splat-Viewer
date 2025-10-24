struct VertexInput
{
    @location(0) pos: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos: vec4<f32>
};

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) idx : u32) -> VertexOutput 
{
    var out: VertexOutput;

    let splat = splats[idx];
    let center = splat.pos;

    out.position = center + in.pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}