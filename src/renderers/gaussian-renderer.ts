import { PointCloud } from '../utils/load'
import preprocessWGSL from '../shaders/preprocess.wgsl'
import renderWGSL from '../shaders/gaussian.wgsl'
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort'
import { Renderer } from './renderer'

export interface GaussianRenderer extends Renderer {}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage })
  if (data) device.queue.writeBuffer(buffer, 0, data)
  return buffer
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  vertexBuffer: GPUBuffer,
  indexBuffer: GPUBuffer,
  drawBuffer: GPUBuffer,
  splatBuffer: GPUBuffer
): GaussianRenderer {
  const sorter = get_sorter(pc.num_points, device)

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  // create draw buffer
  // we will fill this in in the compute shader
  const n = pc.num_points

  vertexBuffer = createBuffer(
    device,
    'vertex buffer',
    4 * 4 * 4, // 1 quad (instanced) * 4 vertices * 4 bytes per float * 4 floats per vec4
    GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX
  )

  indexBuffer = createBuffer(
    device,
    'index buffer',
    6 * 4, // 1 point cloud (instanced) * 6 indices * 4 bytes per index
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX
  )

  drawBuffer = createBuffer(
    device,
    'draw buffer',
    5 * 4, // 1 draw command (instanced) * 5 ints per indexed indirect draw call * 4 bytes per int
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
  )

  const splatSize = n * 4 * 4; // n point clouds * 4 bytes per float * 4 floats per vec4<f32>
  splatBuffer = createBuffer(
    device,
    'splat buffer',
    splatSize,
    GPUBufferUsage.STORAGE
  )
    
  const nulling_data = new Uint32Array([0, 0, 0, 0, 0])

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  })

  const drawBindGroup = device.createBindGroup({
    label: 'draw bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: drawBuffer } },
      { binding: 2, resource: { buffer: vertexBuffer } },
      { binding: 3, resource: { buffer: indexBuffer } },
    ],
  })

  const gaussianBindGroup = device.createBindGroup({
    label: 'gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splatBuffer } },
    ],
  })

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      /*
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      {
        binding: 1,
        resource: { buffer: sorter.ping_pong[0].sort_depths_buffer },
      },
      {
        binding: 2,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
      },
      {
        binding: 3,
        resource: { buffer: sorter.sort_dispatch_indirect_buffer },
      },*/
    ],
  })

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const render_shader = device.createShaderModule({ code: renderWGSL })
  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 4 * 4, // vec4
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
        },
      ],
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  })

    const renderBindGroup = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
    ]
  })


  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const computePass = encoder.beginComputePass()
    computePass.setPipeline(preprocess_pipeline)
    computePass.setBindGroup(0, drawBindGroup)
    computePass.setBindGroup(1, gaussianBindGroup)
    //computePass.setBindGroup(2, sort_bind_group)

    const groups = Math.ceil(n / C.histogram_wg_size)

    device.queue.writeBuffer(drawBuffer, 0, nulling_data); // reset instance counts
    computePass.dispatchWorkgroups(groups)

    computePass.end()

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })

    pass.setPipeline(render_pipeline)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.setIndexBuffer(indexBuffer, 'uint32')
    pass.setBindGroup(0, renderBindGroup);

    pass.drawIndexedIndirect(drawBuffer, 0)

    pass.end()
  }

  // ===============================================
  //    Return Render Object
  // ===============================================

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      render(encoder, texture_view)
    },

    camera_buffer,
  }
}
