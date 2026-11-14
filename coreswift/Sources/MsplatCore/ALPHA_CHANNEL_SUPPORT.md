# Alpha Channel Support in MetalPipeline

## Overview
The `RenderParams` struct and related rendering code now support both 3-channel (RGB) and 4-channel (RGBA) rendering.

## Changes Made

### 1. RenderParams Structure
Added a `channels` parameter (default = 3):
```swift
struct RenderParams {
    // ... existing fields ...
    var channels: UInt32 = 3  // 3 for RGB, 4 for RGBA
}
```

### 2. FusedTensorCache
- Added `channels` field to track current channel configuration
- Updated `ensureForward()` to accept and respond to channel count changes
- Updated `ensureChunks()` to allocate chunk buffers with the correct channel count
- Output tensors (`outImg`, `vRendered`) now resize when channel count changes

### 3. Rendering Functions
- `encodeRastFwd()`: Now uses `p.channels` instead of hardcoded `3`
- `msplatRender()`: Passes channel count to cache allocation functions
- `msplatTrainStep()`: Passes channel count to cache allocation functions

## Usage

### Rendering with RGB (default)
```swift
var params = RenderParams(
    numPoints: gaussians.count,
    // ... other parameters ...
    channels: 3  // or omit for default
)
let outputImage = msplatRender(params)
// outputImage shape: [height, width, 3]
```

### Rendering with RGBA
```swift
var params = RenderParams(
    numPoints: gaussians.count,
    // ... other parameters ...
    channels: 4
)
let outputImage = msplatRender(params)
// outputImage shape: [height, width, 4]
```

## Important Notes

1. **Metal Shader Compatibility**: The Metal shaders (in `msplat_metal.metal`) must also be updated to handle 4 channels. The current shaders likely hardcode RGB (3 channels).

2. **Background Tensor**: When using 4 channels, ensure the `background` tensor in `RenderParams` also has 4 channels (RGBA) instead of 3.

3. **Training**: Training typically uses 3 channels for loss computation (RGB only). If you need to train with alpha, additional loss computation updates may be required.

4. **Memory Impact**: Using 4 channels increases memory usage by ~33% for output buffers:
   - `outImg`: [H, W, 3] → [H, W, 4]
   - `vRendered`: [H, W, 3] → [H, W, 4]
   - Chunked buffers: `chunkC` and `afterC` also scale with channels

5. **Performance**: The additional channel may slightly impact performance due to increased memory bandwidth requirements.

## Next Steps

To fully support RGBA rendering, you'll also need to update:

1. **Metal Shaders** (`msplat_metal.metal`):
   - `nd_rasterize_forward`: Update to write 4 channels
   - `rasterize_forward_chunked`: Update to accumulate 4 channels
   - `rasterize_forward_merge`: Update to merge 4 channels
   - `rasterize_backward` kernels: Update to handle 4-channel gradients

2. **Loss Functions** (if training with alpha):
   - `ssim_h_fwd`, `ssim_fused_v_fwd_h_bwd`, `ssim_v_bwd`
   - May need separate handling for RGB vs RGBA loss computation

3. **Opacity Handling**:
   - Decide whether alpha should be derived from gaussian opacities or as a separate channel
   - Update blending logic in rasterization kernels accordingly
