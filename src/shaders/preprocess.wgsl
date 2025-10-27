const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    pad_0: f32,
    pad_1: f32
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    pos: u32,
    size: u32,
    col_op: vec2u,
    conic: vec2u
};

// splat bind group
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats : array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_buff: array<u32>;


//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> 
{
    let index = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;
    let color01 = unpack2x16float(sh_buff[index + 0]);
    let color23 = unpack2x16float(sh_buff[index + 1]);

    if (c_idx % 2 == 0) 
    {
        return vec3f(color01.x, color01.y, color23.x);
    }

    return vec3f(color01.y, color23.x, color23.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;

    if (idx >= arrayLength(&gaussians))
    {
        return;
    }

    let gaussian = gaussians[idx];

    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec3f(a.xy, b.x);
    let view = camera.view * vec4<f32>(pos.xyz, 1.0f);
    let clip = camera.proj * view;
    let center = clip.xyzw / clip.w;

    let aabbLim = 2.2f;
    if (abs(center.x) > aabbLim || abs(center.y) > aabbLim)
    {
        return;
    }

    // 3D covariance
    let rot = vec4<f32>(unpack2x16float(gaussian.rot[0]), unpack2x16float(gaussian.rot[1]));
    let scale = exp(vec3f(unpack2x16float(gaussian.scale[0]), unpack2x16float(gaussian.scale[1]).x)) * settings.gaussian_scaling;

    let R = mat3x3f(
        1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z + rot.x * rot.w), 2.0 * (rot.y * rot.w - rot.x * rot.z),
        2.0 * (rot.y * rot.z - rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w + rot.x * rot.y),
        2.0 * (rot.y * rot.w + rot.x * rot.z), 2.0 * (rot.z * rot.w - rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
    );

    let S = mat3x3f(
        scale.x, 0.0f, 0.0f,
        0.0f, scale.y, 0.0f,
        0.0f, 0.0f, scale.z
    );

    let cov3D = R * S * transpose(S) * transpose(R);

    // 2D covariance
    let W = mat3x3f(
        vec3f(camera.view[0][0], camera.view[1][0], camera.view[2][0]),
        vec3f(camera.view[0][1], camera.view[1][1], camera.view[2][1]),
        vec3f(camera.view[0][2], camera.view[1][2], camera.view[2][2])
    );

    let invZ = 1.0 / view.z;
    let J = mat2x3f(
        camera.focal.x * invZ, 0.0, -camera.focal.x * view.x * invZ * invZ,
        0.0, camera.focal.y * invZ, -camera.focal.y * view.y * invZ * invZ
    );

    let M = W * cov3D * transpose(W); 
    var cov2D = (transpose(J) * M) * J;

    // numerical stability
    cov2D[0][0] += 0.3f;
    cov2D[1][1] += 0.3f;

    // compute splat paramss
    let det = determinant(cov2D);
    if (det == 0)
    {
        return;
    }

    let mid = 0.5 * (cov2D[0][0] + cov2D[1][1]);
	let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

    let r = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    let radius = vec2f(r) / camera.viewport;

    // compute color
    let camPos = -camera.view[3].xyz;
    let direction = normalize(pos - camPos);

    let color = computeColorFromSH(
        direction, idx, u32(settings.sh_deg)
    );
    let opacity = 1.0f / (1.0f + exp(-b.y)); // sigmoid

    let sortIdx = atomicAdd(&sort_infos.keys_size, 1);
    sort_depths[sortIdx] = bitcast<u32>(100.0f - view.z);
    sort_indices[sortIdx] = sortIdx; // populate with index for radix

    let debug = sort_dispatch.dispatch_y;
    splats[sortIdx].pos = pack2x16float(center.xy);
    splats[sortIdx].size = pack2x16float(radius);
    splats[sortIdx].col_op = vec2u(
        pack2x16float(color.rg),
        pack2x16float(vec2f(color.b, opacity))
    );

    let det_inv = 1.0 / det;
    let conic = vec3f(
        cov2D[1][1] * det_inv,
        -cov2D[0][1] * det_inv,
        cov2D[0][0] * det_inv
    );
    splats[sortIdx].conic = vec2u(
        pack2x16float(conic.xy),
        pack2x16float(vec2f(conic.z, opacity))
    );

    // sort dispatch
    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 

    if (sortIdx % keys_per_dispatch == 0) 
    {
        atomicAdd(&sort_dispatch.dispatch_x, 1u); // add representative thread for dispatch group
    }
}