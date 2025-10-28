import { PointCloud } from '../utils/load'
import preprocessWGSL from '../shaders/preprocess.wgsl'
import renderWGSL from '../shaders/gaussian.wgsl'
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort'
import { Renderer } from './renderer'

export interface GaussianRenderer extends Renderer {
  settingsBuffer: GPUBuffer
}

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
  drawBuffer: GPUBuffer,
  splatBuffer: GPUBuffer,
  settingsBuffer: GPUBuffer,
  nullingBuffer: GPUBuffer
): GaussianRenderer {
  const sorter = get_sorter(pc.num_points, device)

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  // create draw buffer
  // we will fill this in in the compute shader
  const n = pc.num_points

  drawBuffer = createBuffer(
    device,
    'draw buffer',
    4 * 4, // 1 draw command (instanced) * 4 ints per indexed indirect draw call * 4 bytes per int
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([4, 0, 0, 0])
  )

  const splatSize = n * 6 * 4 // n point clouds * 6 packed floats * 4 bytes per packed float
  splatBuffer = createBuffer(
    device,
    'splat buffer',
    splatSize,
    GPUBufferUsage.STORAGE
  )

  settingsBuffer = createBuffer(
    device,
    'settings buffer',
    4 * 4, // 2 floats + padding
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1, pc.sh_deg, 0, 0])
  )

  const nulling_data = new Uint32Array([0, 1, 1])
  nullingBuffer = createBuffer(
    device,
    'nulling buffer',
    12,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    nulling_data
  )

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
      { binding: 1, resource: { buffer: settingsBuffer } },
    ],
  })

  const gaussianBindGroup = device.createBindGroup({
    label: 'gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splatBuffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  })

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
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
      },
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
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [
        {
          format: presentation_format,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  })

  const renderBindGroup = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
      {
        binding: 1,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
      },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  })

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const preprocess = (encoder: GPUCommandEncoder) => {
    const computePass = encoder.beginComputePass()
    computePass.setPipeline(preprocess_pipeline)
    computePass.setBindGroup(0, drawBindGroup)
    computePass.setBindGroup(1, gaussianBindGroup)
    computePass.setBindGroup(2, sort_bind_group)

    const groups = Math.ceil(n / C.histogram_wg_size)

    computePass.dispatchWorkgroups(groups)

    computePass.end()
  }

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
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
    pass.setBindGroup(0, renderBindGroup)

    pass.drawIndirect(drawBuffer, 0)

    pass.end()
  }

  // ===============================================
  //    Return Render Object
  // ===============================================

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      encoder.copyBufferToBuffer(
        nullingBuffer,
        0,
        sorter.sort_info_buffer,
        0,
        4
      )
      encoder.copyBufferToBuffer(
        nullingBuffer,
        0,
        sorter.sort_dispatch_indirect_buffer,
        0,
        12
      )

      preprocess(encoder)

      /*
        const test = createBuffer(
        device,
        'test buffer',
        3 * 4, // 2 floats + padding
        GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        new Float32Array([n, 1, 1])
      )
      encoder.copyBufferToBuffer(test, 0, sorter.sort_dispatch_indirect_buffer, 0, 12);*/
      
      sorter.sort(encoder)

      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, drawBuffer, 4, 4) // extract instance count from sort buffer

      render(encoder, texture_view)
    },

    camera_buffer,
    settingsBuffer,
  }
}
