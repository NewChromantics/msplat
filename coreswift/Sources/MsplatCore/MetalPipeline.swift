// Port of the dispatch half of metal/msplat_metal.mm:
//   msplat_render, msplat_train_step (fused fwd + bwd + Adam + grad stats) and
//   msplat_densify (GPU classify → grow → cull → compact).
// Buffer indices, scalar types and dispatch sizes match the C++ exactly; the shader
// (msplat_metal.metal) is unchanged.

import Foundation
import Metal
import simd
import PopCommon


private let RAST_BLOCK_X = 8
private let RAST_BLOCK_Y = 8
/// Must match MAX_TILE_ELEMS in msplat_metal.metal.
private let MAX_TILE_ELEMS = 2048
private let CHUNK_SIZE: UInt32 = 512
private let BWD_CHUNK_SIZE: UInt32 = 512

private let benchmarkEnabled = env("BENCHMARK")

// MARK: - Encoder helpers (ENC_SCALAR / ENC_ARRAY / ENC_BUF)

extension MTLComputeCommandEncoder {
    @inline(__always)
    func setScalar<T>(_ v: T, _ index: Int) {
        withUnsafeBytes(of: v) { bytes in
            setBytes(bytes.baseAddress!, length: bytes.count, index: index)
        }
    }

    @inline(__always)
    func setArray<T>(_ a: [T], _ index: Int) {
        a.withUnsafeBufferPointer { buffer in
            setBytes(buffer.baseAddress!, length: MemoryLayout<T>.stride * buffer.count, index: index)
        }
    }

    @inline(__always)
    func setTensor(_ t: MTensor, _ index: Int) {
        setBuffer(t.buffer, offset: 0, index: index)
    }

    @inline(__always)
    func barrier() { memoryBarrier(scope: .buffers) }

    @inline(__always)
    func dispatch1D(_ n: Int, _ pso: MTLComputePipelineState) {
        let tpg = min(pso.maxTotalThreadsPerThreadgroup, n)
        dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: max(tpg, 1), height: 1, depth: 1))
    }
}

extension MTLBlitCommandEncoder {
    func zero(_ t: MTensor) {
        guard let b = t.buffer else { return }
        fill(buffer: b, range: 0..<t.nbytes, value: 0)
    }
}

// MARK: - Shared parameters

/// Camera intrinsics parameters
public struct CameraIntrinsics : Sendable 
{
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float
    public var width: Int
    public var height: Int
    
    public var imgWidth: UInt32 { UInt32(width) }
    public var imgHeight: UInt32 { UInt32(height) }
    
    /// Horizontal field of view in radians
    public var fovX: Float {
        2.0 * atanf(Float(width) / (2.0 * fx))
    }
    
    /// Vertical field of view in radians
    public var fovY: Float {
        2.0 * atanf(Float(height) / (2.0 * fy))
    }
    
    public init(fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    }
    
    /// Scale intrinsics by a factor (e.g., for downsampling)
    public func scaled(by factor: Float) -> CameraIntrinsics {
        let invFactor = 1.0 / factor
        return CameraIntrinsics(
            fx: fx * invFactor,
            fy: fy * invFactor,
            cx: cx * invFactor,
            cy: cy * invFactor,
            width: Int(Float(width) * invFactor),
            height: Int(Float(height) * invFactor)
        )
    }
}


public struct SplatBuffers
{
	var means3d : MTensor
	var scales : MTensor
	var quats : MTensor
	var featuresDc : MTensor
	var featuresRest : MTensor
	var opacities : MTensor
}

// Per-view camera/render parameters shared by render and train step.
public struct RenderParams 
{
	var numPoints: Int
	var splatBuffers : SplatBuffers
	var globScale: Float
	var viewmat: simd_float4x4
	var projmat: simd_float4x4
	var camPos: simd_float3
	var intrinsics: CameraIntrinsics
	var tileBounds: (Int, Int, Int)
	var clipThresh: Float
	var degree: UInt32
	var degreesToUse: UInt32
	var background: simd_float3
	var channels: UInt32  // 3 for RGB, 4 for RGBA
}

/// Training-only inputs to the fused step.
public struct TrainParams {
    var gt: MTensor
    var window2d: MTensor
    var ssimWeight: Float
    var lossInvN: Float
    var featuresRestBases: Int
    var adamParams: [MTensor]       // 6 groups
    var adamExpAvg: [MTensor]
    var adamExpAvgSq: [MTensor]
    var adamStepSizes: [Float]
    var adamBc2Sqrts: [Float]
    var adamBeta1: Float, adamBeta2: Float, adamEps: Float
    var visCounts: MTensor
    var xysGradNorm: MTensor
    var max2dSize: MTensor
    var invMaxDim: Float
}

/// Packed SH Adam hyperparameters (must match SHAdamParams in .metal: 7 floats).
private struct SHAdamParams {
    var dc_step_size: Float = 0, dc_bc2_sqrt: Float = 0
    var rest_step_size: Float = 0, rest_bc2_sqrt: Float = 0
    var beta1: Float = 0, beta2: Float = 0, eps: Float = 0
}

/// Constant blocks bound with setBytes, derived once per call.
private struct Consts {
    let tileBoundsX: Int, tileBoundsY: Int, numTiles: Int
    let capacity: Int
    let numPointsU32: UInt32
    let projIntrins: [Float]       // float4
    let imgSize2: [UInt32]         // uint2 {w, h}
    let tileBoundsArr: [UInt32]    // uint3 (+pad)
    let camPos4: [Float]           // float4
    let imgSizeDim3: [UInt32]      // {w, h, 1, pad}
    let blockSizeDim2: [Int32]     // {8, 8}

    init(_ p: RenderParams) {
        tileBoundsX = p.tileBounds.0
        tileBoundsY = p.tileBounds.1
        numTiles = tileBoundsX * tileBoundsY
        // scatter_to_prealloc_bins clamps every tile counter to MAX_TILE_ELEMS, so the
        // prefix sum over them cannot exceed num_tiles * MAX_TILE_ELEMS.
        capacity = numTiles * MAX_TILE_ELEMS
        numPointsU32 = UInt32(p.numPoints)
        projIntrins = [p.intrinsics.fx, p.intrinsics.fy, p.intrinsics.cx, p.intrinsics.cy]
        imgSize2 = [p.intrinsics.imgWidth, p.intrinsics.imgHeight]
        tileBoundsArr = [UInt32(tileBoundsX), UInt32(tileBoundsY), UInt32(p.tileBounds.2), 0xDEAD]
        camPos4 = [p.camPos.x, p.camPos.y, p.camPos.z, 1]
        imgSizeDim3 = [p.intrinsics.imgWidth, p.intrinsics.imgHeight, 1, 0xDEAD]
        blockSizeDim2 = [Int32(RAST_BLOCK_X), Int32(RAST_BLOCK_Y)]
    }
}

/// K_max for chunked rasterization, computed conservatively on the CPU (no sync).
private func computeKMax(numTiles: Int, capacity: Int) -> UInt32 {
    if numTiles >= 400 { return 1 }  // enough tiles for good occupancy: skip chunking
    let avgPerTile = UInt32(capacity / max(1, numTiles))
    let conservativeMax = avgPerTile * 6  // 6x average covers heavy-tailed distributions
    var k = (conservativeMax + CHUNK_SIZE - 1) / CHUNK_SIZE
    if k < 2 { k = 2 }
    // A tile's range is capped at MAX_TILE_ELEMS, so further chunks are always empty.
    let absMax = (UInt32(MAX_TILE_ELEMS) + CHUNK_SIZE - 1) / CHUNK_SIZE
    if k > absMax { k = absMax }
    return k
}

// MARK: - Shared forward-stage encoders

private func encodeProjSh(_ enc: MTLComputeCommandEncoder, _ pso: RenderPipelineStates,
                          _ p: RenderParams, _ k: Consts, _ cache: FusedTensorCache) 
{
    enc.setComputePipelineState(pso.project_and_sh_forward)
    enc.setScalar(Int32(p.numPoints), 0)
	enc.setTensor(p.splatBuffers.means3d, 1)
	enc.setTensor(p.splatBuffers.scales, 2)
    enc.setScalar(p.globScale, 3)
	enc.setTensor(p.splatBuffers.quats, 4)
    // Pass matrices directly as bytes instead of MTensor buffers
    var viewmat = p.viewmat
    var projmat = p.projmat
    enc.setBytes(&viewmat, length: MemoryLayout<simd_float4x4>.size, index: 5)
    enc.setBytes(&projmat, length: MemoryLayout<simd_float4x4>.size, index: 6)
    enc.setArray(k.projIntrins, 7)
    enc.setArray(k.imgSize2, 8)
    enc.setArray(k.tileBoundsArr, 9)
    enc.setScalar(p.clipThresh, 10)
    enc.setTensor(cache.xys, 11)
	enc.setTensor(cache.depths, 12)
    enc.setTensor(cache.radiiOut, 13)
	enc.setTensor(cache.conics, 14)
    enc.setTensor(cache.numTilesHit, 15)
    enc.setScalar(p.degree, 16)
	enc.setScalar(p.degreesToUse, 17)
    enc.setArray(k.camPos4, 18)
	enc.setTensor(p.splatBuffers.featuresDc, 19)
	enc.setTensor(p.splatBuffers.featuresRest, 20)
    enc.setTensor(cache.colors, 21)
	enc.setTensor(cache.aabb, 22)
    // buffer 23 removed (was opacity-aware AABB, reverted)
    enc.dispatch1D(p.numPoints, pso.project_and_sh_forward)
}

private func encodePrefixMap(_ enc: MTLComputeCommandEncoder, _ pso: RenderPipelineStates,
                             _ p: RenderParams, _ k: Consts, _ cache: FusedTensorCache) {
    let numTilesU32 = UInt32(k.numTiles)
    // 1. scatter_to_prealloc_bins
    enc.setComputePipelineState(pso.scatter_to_prealloc_bins)
    enc.setScalar(k.numPointsU32, 0); enc.setTensor(cache.xys, 1); enc.setTensor(cache.depths, 2)
    enc.setTensor(cache.radiiOut, 3); enc.setTensor(cache.aabb, 4)
    enc.setArray(k.tileBoundsArr, 5)
    enc.setTensor(cache.tileScatterCounters, 6)
    enc.setTensor(cache.preallocBins, 7)
    enc.setTensor(cache.overflowFlag, 8)
    enc.dispatch1D(p.numPoints, pso.scatter_to_prealloc_bins)
    enc.barrier()
    // 2. prefix_sum(tile_scatter_counters -> tile_offsets), single threadgroup
    let tg2 = min(pso.prefix_sum.maxTotalThreadsPerThreadgroup, 1024)
    enc.setComputePipelineState(pso.prefix_sum)
    enc.setScalar(numTilesU32, 0); enc.setTensor(cache.tileScatterCounters, 1); enc.setTensor(cache.tileOffsets, 2)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
    enc.barrier()
    // 3. bitonic_sort_per_tile (reads prealloc_bins, writes packed + tile_bins)
    enc.setComputePipelineState(pso.bitonic_sort_per_tile)
    enc.setTensor(cache.tileOffsets, 0); enc.setTensor(cache.tileScatterCounters, 1)
    enc.setTensor(cache.preallocBins, 2)
    enc.setTensor(cache.gaussianIds, 3)
    enc.setScalar(numTilesU32, 4)
    enc.setTensor(cache.xys, 5); enc.setTensor(cache.conics, 6)
	enc.setTensor(cache.colors, 7); enc.setTensor(p.splatBuffers.opacities, 8)
    enc.setTensor(cache.packedXyOpac, 9); enc.setTensor(cache.packedConic, 10); enc.setTensor(cache.packedRgb, 11)
    enc.setTensor(cache.tileBins, 12)
    enc.dispatchThreadgroups(MTLSize(width: k.numTiles, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

private func encodeRastFwd(_ enc: MTLComputeCommandEncoder, _ pso: RenderPipelineStates,
                           _ p: RenderParams, _ k: Consts, kMax: UInt32, _ cache: FusedTensorCache) 
{
    let channels = p.channels
    let w = p.intrinsics.width, h = p.intrinsics.height
    let tileX = (w + RAST_BLOCK_X - 1) / RAST_BLOCK_X
    let tileY = (h + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y
    let tg = MTLSize(width: RAST_BLOCK_X, height: RAST_BLOCK_Y, depth: 1)

    if kMax <= 1 {
        // Monolithic
		var background = p.background
        enc.setComputePipelineState(pso.nd_rasterize_forward)
        enc.setArray(k.tileBoundsArr, 0)
        enc.setArray(k.imgSizeDim3, 1)
        enc.setScalar(channels, 2)
		enc.setTensor(cache.tileBins, 3)
        enc.setTensor(cache.packedXyOpac, 4)
		enc.setTensor(cache.packedConic, 5)
		enc.setTensor(cache.packedRgb, 6)
        enc.setTensor(cache.finalTs, 7)
		enc.setTensor(cache.finalIdx, 8)
		enc.setTensor(cache.outImg, 9)
		enc.setBytes(&background, index: 10)
        enc.setArray(k.blockSizeDim2, 11)
        enc.dispatchThreadgroups(MTLSize(width: tileX, height: tileY, depth: 1), threadsPerThreadgroup: tg)
    }
	else 
	{
		fatalError("never used?")
        // Phase 1: chunked forward — grid (tile_x, tile_y, K_max)
        let numPix = UInt32(w * h)
        enc.setComputePipelineState(pso.rasterize_forward_chunked)
        enc.setArray(k.tileBoundsArr, 0)
        enc.setArray(k.imgSizeDim3, 1)
        enc.setScalar(channels, 2)
		enc.setTensor(cache.tileBins, 3)
        enc.setTensor(cache.packedXyOpac, 4)
		enc.setTensor(cache.packedConic, 5)
		enc.setTensor(cache.packedRgb, 6)
        enc.setTensor(cache.chunkT, 7)
		enc.setTensor(cache.chunkC, 8)
		enc.setTensor(cache.chunkFinalIdx, 9)
        enc.setScalar(CHUNK_SIZE, 10)
		enc.setScalar(kMax, 11)
        enc.setArray(k.blockSizeDim2, 12)
        enc.dispatchThreadgroups(MTLSize(width: tileX, height: tileY, depth: Int(kMax)), threadsPerThreadgroup: tg)
        enc.barrier()
        // Phase 2: merge — one thread per pixel
        enc.setComputePipelineState(pso.rasterize_forward_merge)
        enc.setScalar(numPix, 0)
		enc.setScalar(kMax, 1)
        enc.setTensor(cache.chunkT, 2)
		enc.setTensor(cache.chunkC, 3)
		enc.setTensor(cache.chunkFinalIdx, 4)
        enc.setTensor(cache.finalTs, 5)
		enc.setTensor(cache.finalIdx, 6)
		enc.setTensor(cache.outImg, 7)
        enc.setScalar(p.background, 8)
        enc.setArray(k.imgSize2, 9)
        enc.setScalar(channels, 10)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
}

// MARK: - GPU Execution Helper

/// Pipeline states required for rendering. Can be created from any MetalContext.
public struct RenderPipelineStates {
    let project_and_sh_forward: MTLComputePipelineState
    let scatter_to_prealloc_bins: MTLComputePipelineState
    let bitonic_sort_per_tile: MTLComputePipelineState
    let prefix_sum: MTLComputePipelineState
    let nd_rasterize_forward: MTLComputePipelineState
    let rasterize_forward_chunked: MTLComputePipelineState
    let rasterize_forward_merge: MTLComputePipelineState
    
    public init(ctx: MetalContext) {
        self.project_and_sh_forward = ctx.project_and_sh_forward
        self.scatter_to_prealloc_bins = ctx.scatter_to_prealloc_bins
        self.bitonic_sort_per_tile = ctx.bitonic_sort_per_tile
        self.prefix_sum = ctx.prefix_sum
        self.nd_rasterize_forward = ctx.nd_rasterize_forward
        self.rasterize_forward_chunked = ctx.rasterize_forward_chunked
        self.rasterize_forward_merge = ctx.rasterize_forward_merge
    }
}
/*
/// GPU execution statistics returned from executeGpuWork
public struct GpuExecuteStats {
    public var name: String
    public var gpuExecutionDurationMs: Int
    public var gpuScheduleDurationMs: Int
}
*/
extension MTLCommandQueue {
    /// Execute GPU work asynchronously on this command queue.
    /// Returns timing statistics when GPU work is complete.
    @discardableResult
    func executeGpuWork(
        jobName: String = "GPUWork",
        _ work: (MTLCommandBuffer) throws -> Void
    ) async throws -> GpuExecuteStats {
        guard let cb = self.makeCommandBuffer() else {
            throw MsplatError("Failed to create command buffer")
        }
        
        try work(cb)
        
        let promise = SendablePromise<GpuExecuteStats>()
        cb.addCompletedHandler { completionMeta in
            let executeMs = (completionMeta.gpuEndTime - completionMeta.gpuStartTime) * 1000.0
            let scheduleMs = (completionMeta.kernelEndTime - completionMeta.kernelStartTime) * 1000.0
            let stats = GpuExecuteStats(
                name: jobName,
                gpuExecutionDurationMs: Int(executeMs),
                gpuScheduleDurationMs: Int(scheduleMs)
            )
            promise.Resolve(stats)
        }
        cb.commit()
        
        let stats = try await promise.value
        //print("\(jobName) took \(stats.gpuExecutionDurationMs)ms (\(stats.gpuScheduleDurationMs)ms setup)")
        return stats
    }
}

// MARK: - Render (forward only)

/// Buffers required for a render pass. Create once and reuse across multiple renders
/// with the same dimensions.
public struct RenderBuffers {
    // Per-point intermediate buffers
    var xys: MTensor
    var depths: MTensor
    var radiiOut: MTensor
    var conics: MTensor
    var numTilesHit: MTensor
    var colors: MTensor
    var aabb: MTensor
    var blockTotals: MTensor
    
    // Per-tile sorting buffers
    var gaussianIds: MTensor
    var packedXyOpac: MTensor
    var packedConic: MTensor
    var packedRgb: MTensor
    var tileBins: MTensor
    var tileOffsets: MTensor
    var tileScatterCounters: MTensor
    var preallocBins: MTensor
    
    // Output buffers
    var outImg: MTensor
    var finalTs: MTensor
    var finalIdx: MTensor
    
    // Utility buffers
    var lossSum: MTensor
    var overflowFlag: MTensor
    
    // Optional: depth-chunked rasterization buffers (allocated when kMax > 1)
    var chunkT: MTensor?
    var chunkC: MTensor?
    var chunkFinalIdx: MTensor?
    var prefixT: MTensor?
    var afterC: MTensor?
    
    /// Create render buffers for the given dimensions using any Metal device.
    public static func allocate(
        numPoints: Int,
        capacity: Int,
        imageHeight: Int,
        imageWidth: Int,
        numTiles: Int,
        channels: Int,
        kMax: Int = 1,
        device: MTLDevice
    ) throws -> RenderBuffers {
        let np = numPoints
        let cap = capacity
        let ih = imageHeight
        let iw = imageWidth
        let nt = numTiles
        let ch = channels
        
        // Helper to create empty tensors from device
        func makeEmpty(_ shape: [Int], _ dtype: DType = .float32) throws -> MTensor {
            return try MTensor(device: device, shape: shape, dtype: dtype)
        }
        
        var buffers = RenderBuffers(
            xys: try makeEmpty([np, 2], .float32),
            depths: try makeEmpty([np], .float32),
            radiiOut: try makeEmpty([np], .int32),
            conics: try makeEmpty([np, 3], .float32),
            numTilesHit: try makeEmpty([np], .int32),
            colors: try makeEmpty([np, 3], .float32),
            aabb: try makeEmpty([np, 2], .float32),
            blockTotals: try makeEmpty([(np + 1023) / 1024], .int32),
            
            gaussianIds: try makeEmpty([cap], .int32),
            packedXyOpac: try makeEmpty([cap, 3], .float32),
            packedConic: try makeEmpty([cap, 3], .float32),
            packedRgb: try makeEmpty([cap, 3], .float32),
            tileBins: try makeEmpty([nt, 2], .int32),
            tileOffsets: try makeEmpty([nt], .int32),
            tileScatterCounters: try makeEmpty([nt], .int32),
            preallocBins: try makeEmpty([nt * 2048], .int64),
            
            outImg: try makeEmpty([ih, iw, ch], .float32),
            finalTs: try makeEmpty([ih, iw], .float32),
            finalIdx: try makeEmpty([ih, iw], .int32),
            
            lossSum: try makeEmpty([1], .float32),
            overflowFlag: try makeEmpty([1], .int32),
            
            chunkT: nil,
            chunkC: nil,
            chunkFinalIdx: nil,
            prefixT: nil,
            afterC: nil
        )
        
        // Allocate chunked rendering buffers if needed
        if kMax > 1 {
            buffers.chunkT = try makeEmpty([kMax, ih, iw], .float32)
            buffers.chunkC = try makeEmpty([kMax, ih, iw, ch], .float32)
            buffers.chunkFinalIdx = try makeEmpty([kMax, ih, iw], .int32)
            buffers.prefixT = try makeEmpty([kMax, ih, iw], .float32)
            buffers.afterC = try makeEmpty([kMax, ih, iw, ch], .float32)
        }
        
        return buffers
    }
}

/// Refined render function that uses explicit buffer allocation.
/// Buffers can be pre-allocated and reused across multiple render calls.
/// Works with any Metal device and command queue - no MetalContext required.
public func msplatRenderWithBuffers(
    renderParams: RenderParams,
    _ buffers: RenderBuffers,
    queue: MTLCommandQueue,
    pipelineStates: RenderPipelineStates
) async throws -> MTensor 
{
    let k = Consts(renderParams)
    let kMax = computeKMax(numTiles: k.numTiles, capacity: k.capacity)
    
    // Validate that chunked buffers exist if needed
    if kMax > 1 {
        guard buffers.chunkT != nil else {
            throw MsplatError("msplatRenderWithBuffers: kMax=\(kMax) requires chunked buffers")
        }
    }
	
	print("msplatRenderWithBuffers - intrinsics \(renderParams.intrinsics)")
    
    // Create a cache-like struct that wraps the provided buffers
    var cache = FusedTensorCache()
    cache.xys = buffers.xys
    cache.depths = buffers.depths
    cache.radiiOut = buffers.radiiOut
    cache.conics = buffers.conics
    cache.numTilesHit = buffers.numTilesHit
    cache.colors = buffers.colors
    cache.aabb = buffers.aabb
    cache.blockTotals = buffers.blockTotals
    cache.gaussianIds = buffers.gaussianIds
    cache.packedXyOpac = buffers.packedXyOpac
    cache.packedConic = buffers.packedConic
    cache.packedRgb = buffers.packedRgb
    cache.tileBins = buffers.tileBins
    cache.tileOffsets = buffers.tileOffsets
    cache.tileScatterCounters = buffers.tileScatterCounters
    cache.preallocBins = buffers.preallocBins
    cache.outImg = buffers.outImg
    cache.finalTs = buffers.finalTs
    cache.finalIdx = buffers.finalIdx
    cache.lossSum = buffers.lossSum
    cache.overflowFlag = buffers.overflowFlag
    
    if let chunkT = buffers.chunkT 
	{
        cache.chunkT = chunkT
        cache.chunkC = buffers.chunkC!
        cache.chunkFinalIdx = buffers.chunkFinalIdx!
        cache.prefixT = buffers.prefixT!
        cache.afterC = buffers.afterC!
    }
    
    // Execute render and wait for completion
    try await queue.executeGpuWork(jobName: "Render")
    {
        cb in
		
        // Zero accumulating buffers on the GPU
        let blit = cb.makeBlitCommandEncoder()!
        blit.zero(cache.lossSum)
        blit.zero(cache.overflowFlag)
        blit.zero(cache.tileScatterCounters)
        blit.zero(cache.numTilesHit)
        blit.zero(cache.radiiOut)
        blit.endEncoding()

        let enc = cb.makeComputeCommandEncoder()!
        encodeProjSh(enc, pipelineStates, renderParams, k, cache)
        enc.barrier()
        encodePrefixMap(enc, pipelineStates, renderParams, k, cache)
        enc.barrier()
        encodeRastFwd(enc, pipelineStates, renderParams, k, kMax: kMax, cache)
        enc.endEncoding()
    }

    return buffers.outImg
}

/// Render-only forward pass using the internal tensor cache.
/// Returns the cached (H, W, 3) output tensor; its contents are valid after the function returns.
func msplatRender(_ p: RenderParams, _ ctx: MetalContext) async throws -> MTensor 
{
    ctx.renderOverflowCheck.run(numPoints: p.numPoints, cache: ctx.tensorCache)

    let k = Consts(p)
    ctx.tensorCache.ensureForwardPassBuffers(np: p.numPoints, cap: k.capacity, ih: p.intrinsics.height,
                               iw: p.intrinsics.width, nt: k.numTiles, ch: Int(p.channels), ctx: ctx)

 
    let kMax = computeKMax(numTiles: k.numTiles, capacity: k.capacity)
    if kMax > 1
	{
        try ctx.tensorCache.ensureChunkedRenderingBuffers(k: Int(kMax), ih: p.intrinsics.height, iw: p.intrinsics.width, ch: Int(p.channels), ctx: ctx)
    }

    // Convert tensor cache to RenderBuffers struct
    let buffers = RenderBuffers(
        xys: ctx.tensorCache.xys,
        depths: ctx.tensorCache.depths,
        radiiOut: ctx.tensorCache.radiiOut,
        conics: ctx.tensorCache.conics,
        numTilesHit: ctx.tensorCache.numTilesHit,
        colors: ctx.tensorCache.colors,
        aabb: ctx.tensorCache.aabb,
        blockTotals: ctx.tensorCache.blockTotals,
        gaussianIds: ctx.tensorCache.gaussianIds,
        packedXyOpac: ctx.tensorCache.packedXyOpac,
        packedConic: ctx.tensorCache.packedConic,
        packedRgb: ctx.tensorCache.packedRgb,
        tileBins: ctx.tensorCache.tileBins,
        tileOffsets: ctx.tensorCache.tileOffsets,
        tileScatterCounters: ctx.tensorCache.tileScatterCounters,
        preallocBins: ctx.tensorCache.preallocBins,
        outImg: ctx.tensorCache.outImg,
        finalTs: ctx.tensorCache.finalTs,
        finalIdx: ctx.tensorCache.finalIdx,
        lossSum: ctx.tensorCache.lossSum,
        overflowFlag: ctx.tensorCache.overflowFlag,
        chunkT: kMax > 1 ? ctx.tensorCache.chunkT : nil,
        chunkC: kMax > 1 ? ctx.tensorCache.chunkC : nil,
        chunkFinalIdx: kMax > 1 ? ctx.tensorCache.chunkFinalIdx : nil,
        prefixT: kMax > 1 ? ctx.tensorCache.prefixT : nil,
        afterC: kMax > 1 ? ctx.tensorCache.afterC : nil
    )

    // Create pipeline states from context
    let pso = RenderPipelineStates(ctx: ctx)

    // Delegate to the refined render function
	return try await msplatRenderWithBuffers(renderParams:p, buffers, queue: ctx.queue, pipelineStates: pso)
}


func GetSplatBuffers(splats:[PlySplatPoint],device:MTLDevice) throws -> SplatBuffers
{
	let numSplats = splats.count
	
	guard numSplats > 0 else {
		throw MsplatError("GetSplatBuffers: no splats provided")
	}
	
	// Create GPU tensors from splat data
	let means3d = try MTensor(device: device, shape: [numSplats, 3], dtype: .float32)
	let scales = try MTensor(device: device, shape: [numSplats, 3], dtype: .float32)
	let quats = try MTensor(device: device, shape: [numSplats, 4], dtype: .float32)
	let featuresDc = try MTensor(device: device, shape: [numSplats, 3], dtype: .float32)
	let opacities = try MTensor(device: device, shape: [numSplats, 1], dtype: .float32)
	
	// Upload splat data using SIMD accessors
	var means3dPtr = means3d.floats()
	var scalesPtr = scales.floats()
	var quatsPtr = quats.floats()
	var featuresDcPtr = featuresDc.floats()
	let opacitiesPtr = opacities.floats()
	
	for i in 0..<numSplats 
	{
		let s = splats[i]
		// Position
		means3dPtr[float3: i] = s.position
		// Scale (log space)
		scalesPtr[float3: i] = s.scaleLog
		// Quaternion (w, x, y, z)
		quatsPtr[float4: i] = s.quaternionWxyz
		// SH DC term
		featuresDcPtr[float3: i] = s.shDc
		// Opacity (logit)
		opacitiesPtr[i] = s.opacityLogit
	}
	
	// Create empty tensor for higher-order SH (we're only using DC term)
	// Note: Can't create tensor with 0 dimension, so create minimal [numSplats, 1, 3] but won't be used
	let featuresRest = try MTensor(device: device, shape: [numSplats, 1, 3], dtype: .float32)
	
	let splatBuffers = SplatBuffers(
		means3d: means3d, 
		scales: scales, 
		quats: quats, 
		featuresDc: featuresDc, 
		featuresRest: featuresRest, 
		opacities: opacities
	)
	
	return splatBuffers
}

/// Convenience function to render splat points directly with camera parameters.
/// Creates necessary buffers and renders the scene from the given camera pose.
public func msplatRenderSplats(
    splats: [PlySplatPoint],
    device: MTLDevice,
    queue: MTLCommandQueue,
    pipelineStates: RenderPipelineStates,
	cameraRenderParams:CameraRenderParams,
    intrinsics: CameraIntrinsics,
    backgroundColor: simd_float3,
	format:RGBImage.ImageFormat
) async throws -> MTensor 
{
	let numSplats = splats.count
	
	guard numSplats > 0 else {
		throw MsplatError("msplatRenderSplats: no splats provided")
	}
	
	let splatBuffers = try GetSplatBuffers(splats: splats, device: device)
	
	let renderParams = RenderParams(
		numPoints: numSplats,
		splatBuffers:splatBuffers,
		globScale: 1.0,
		viewmat: cameraRenderParams.viewMat, 
		projmat: cameraRenderParams.projViewMat,
		camPos: cameraRenderParams.camPos, 
		intrinsics: cameraRenderParams.intrinsics,
		tileBounds: cameraRenderParams.tileBounds,
		clipThresh: 0.01,
		degree: UInt32(cameraRenderParams.degrees),
		degreesToUse: UInt32(cameraRenderParams.degreesToUse),
		background: backgroundColor,
		channels: UInt32(format.channels)
	)

    // Calculate buffer requirements
    let k = Consts(renderParams)
    let kMax = Int(computeKMax(numTiles: k.numTiles, capacity: k.capacity))
    
    // Allocate render buffers (these are intermediate computation buffers,
    // not the input splat data which is already in renderParams)
    let buffers = try RenderBuffers.allocate(
        numPoints: numSplats,
        capacity: k.capacity,
        imageHeight: intrinsics.height,
        imageWidth: intrinsics.width,
        numTiles: k.numTiles,
		channels: format.channels,
        kMax: kMax,
        device: device
    )
    
    // Render!
    return try await msplatRenderWithBuffers(
		renderParams:renderParams,
        buffers,
        queue: queue,
        pipelineStates: pipelineStates
    )
}

// MARK: - Fused training step

/// Fused forward + loss + backward + Adam + grad-stats in one command buffer.
/// Returns (radii, loss). The loss value reflects the current step after GPU completion.
func msplatTrainStep(_ p: RenderParams, _ t: TrainParams, _ ctx: MetalContext) async throws -> (radii: MTensor, loss: Float) {
    ctx.trainOverflowCheck.run(numPoints: p.numPoints, cache: ctx.tensorCache)

    let k = Consts(p)
    let w = p.intrinsics.width, h = p.intrinsics.height
    
    // Create pipeline states from context
    let pso = RenderPipelineStates(ctx: ctx)

    // Periodic diagnostic on the PREVIOUS iteration's counters — must run before
    // ensureForward/ensureBackward, which reallocate whenever sizes change.
    ctx.trainDiagIt += 1
    if [200, 1000, 2500, 4500, 6500].contains(ctx.trainDiagIt)
        && ctx.tensorCache.tileScatterCounters.defined && ctx.tensorCache.fwdNumPoints > 0
        && benchmarkEnabled 
	{
        // Sync to read diagnostic data
       // try await ctx.executeAsync { _ in }
        let c = ctx.tensorCache
        let counts = c.tileScatterCounters.int32s()
        let ntPrev = c.tileScatterCounters.numel
        var total = 0, maxc: Int32 = 0
        for i in 0..<ntPrev { total += Int(counts[i]); maxc = max(maxc, counts[i]) }
        let rad = c.radiiOut.int32s()
        let npPrev = c.fwdNumPoints
        var vis = 0
        for i in 0..<npPrev where rad[i] > 0 { vis += 1 }
        logErr(String(format: "[diag it=%ld] %ldx%ld  N=%ld  visible=%ld (%.1f%%)  intersections=%ld"
                      + "  per-vis-gaussian=%.2f  per-tile avg=%.0f max=%ld",
                      ctx.trainDiagIt, c.imgWidth, c.imgHeight, npPrev, vis,
                      100.0 * Double(vis) / Double(max(1, npPrev)), total,
                      Double(total) / Double(max(1, vis)),
                      Double(total) / Double(max(1, ntPrev)), Int(maxc)))
    }

    ctx.tensorCache.ensureForwardPassBuffers(np: p.numPoints, cap: k.capacity, ih: h, iw: w, nt: k.numTiles, ch: Int(p.channels), ctx: ctx)
    ctx.tensorCache.ensureBackwardPassBuffers(np: p.numPoints, frb: t.featuresRestBases, ctx: ctx)

    let kMax = computeKMax(numTiles: k.numTiles, capacity: k.capacity)
    if kMax > 1 {
        try ctx.tensorCache.ensureChunkedRenderingBuffers(k: Int(kMax), ih: h, iw: w, ch: Int(p.channels), ctx: ctx)
    }
    let bwdKMax = kMax

    let c = ctx.tensorCache  // snapshot after (re)allocation; tensors share storage
    let lossImgSize: [UInt32] = [p.intrinsics.imgWidth, p.intrinsics.imgHeight]
    // Rasterize kernels take 16x16 tile counts (tile_bins granularity)
    let rastTb: [UInt32] = [(p.intrinsics.imgWidth + 15) / 16, (p.intrinsics.imgHeight + 15) / 16, 1, 0xDEAD]
    let rastIsz: [UInt32] = [p.intrinsics.imgWidth, p.intrinsics.imgHeight]

    // Backward outputs used as Adam gradients
    let adamGrads = [c.vMean3d, c.vScale, c.vQuat, c.vFeaturesDc, c.vFeaturesRest, c.vOpacity]

    var shAdam = SHAdamParams()
    let numAdamGroups = t.adamParams.count
    if numAdamGroups >= 5 {
        shAdam.dc_step_size = t.adamStepSizes[3]
        shAdam.dc_bc2_sqrt = t.adamBc2Sqrts[3]
        shAdam.rest_step_size = t.adamStepSizes[4]
        shAdam.rest_bc2_sqrt = t.adamBc2Sqrts[4]
        shAdam.beta1 = t.adamBeta1
        shAdam.beta2 = t.adamBeta2
        shAdam.eps = t.adamEps
    }

    // Fused loss: ssim_h_fwd → fused_v_fwd_h_bwd → ssim_v_bwd
    func encodeLossFwdBwd(_ enc: MTLComputeCommandEncoder) {
        let grid = MTLSize(width: w, height: h, depth: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        // Pass 1: H conv on images → ssim_h_buf
        enc.setComputePipelineState(ctx.ssim_h_fwd)
        enc.setTensor(c.outImg, 0); enc.setTensor(t.gt, 1)
        enc.setArray(lossImgSize, 2)
        enc.setTensor(c.ssimHBuf, 3)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.barrier()
        // Pass 2: fused V fwd + H bwd
        enc.setComputePipelineState(ctx.ssim_fused_v_fwd_h_bwd)
        enc.setTensor(c.outImg, 0); enc.setTensor(t.gt, 1)
        enc.setTensor(c.ssimHBuf, 2)
        enc.setArray(lossImgSize, 3)
        enc.setScalar(t.ssimWeight, 4); enc.setScalar(t.lossInvN, 5)
        enc.setTensor(c.lossIntermediates, 6); enc.setTensor(c.lossSum, 7)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.barrier()
        // Pass 3: V bwd
        enc.setComputePipelineState(ctx.ssim_v_bwd)
        enc.setTensor(c.outImg, 0); enc.setTensor(t.gt, 1)
        enc.setTensor(c.lossIntermediates, 2)
        enc.setArray(lossImgSize, 3)
        enc.setScalar(t.ssimWeight, 4); enc.setScalar(t.lossInvN, 5)
        enc.setTensor(c.vRendered, 6)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    }

    func encodeRastBwd(_ enc: MTLComputeCommandEncoder) {
        let tileX = (w + RAST_BLOCK_X - 1) / RAST_BLOCK_X
        let tileY = (h + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y
        let tg = MTLSize(width: RAST_BLOCK_X, height: RAST_BLOCK_Y, depth: 1)
        if bwdKMax <= 1 {
            enc.setComputePipelineState(ctx.rasterize_backward)
            enc.setArray(rastTb, 0)
            enc.setArray(rastIsz, 1)
            enc.setTensor(c.gaussianIds, 2); enc.setTensor(c.tileBins, 3)
            enc.setTensor(c.packedXyOpac, 4); enc.setTensor(c.packedConic, 5)
            enc.setTensor(c.packedRgb, 6)
            enc.setScalar(p.background, 7); enc.setTensor(c.finalTs, 8)
            enc.setTensor(c.finalIdx, 9); enc.setTensor(c.vRendered, 10)
            enc.setTensor(c.vXy, 11); enc.setTensor(c.vConic, 12)
            enc.setTensor(c.vColorsRast, 13); enc.setTensor(c.vOpacity, 14)
            enc.dispatchThreadgroups(MTLSize(width: tileX, height: tileY, depth: 1), threadsPerThreadgroup: tg)
        } else {
            let numPix = UInt32(w * h)
            // Phase 1: prefix_T and after_C
            enc.setComputePipelineState(ctx.compute_chunk_prefix_suffix)
            enc.setScalar(numPix, 0); enc.setScalar(bwdKMax, 1)
            enc.setTensor(c.chunkT, 2); enc.setTensor(c.chunkC, 3)
            enc.setTensor(c.chunkFinalIdx, 4)
            enc.setTensor(c.prefixT, 5); enc.setTensor(c.afterC, 6)
            enc.setArray(rastIsz, 7)
            enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.barrier()
            // Phase 2: chunked backward
            enc.setComputePipelineState(ctx.rasterize_backward_chunked)
            enc.setArray(rastTb, 0)
            enc.setArray(rastIsz, 1)
            enc.setTensor(c.gaussianIds, 2); enc.setTensor(c.tileBins, 3)
            enc.setTensor(c.packedXyOpac, 4); enc.setTensor(c.packedConic, 5)
            enc.setTensor(c.packedRgb, 6)
            enc.setScalar(p.background, 7); enc.setTensor(c.finalTs, 8)
            enc.setTensor(c.chunkFinalIdx, 9)
            enc.setTensor(c.prefixT, 10); enc.setTensor(c.chunkT, 11)
            enc.setTensor(c.afterC, 12)
            enc.setTensor(c.vRendered, 13)
            enc.setTensor(c.vXy, 14); enc.setTensor(c.vConic, 15)
            enc.setTensor(c.vColorsRast, 16); enc.setTensor(c.vOpacity, 17)
            enc.setScalar(BWD_CHUNK_SIZE, 18); enc.setScalar(bwdKMax, 19)
            enc.dispatchThreadgroups(MTLSize(width: tileX, height: tileY, depth: Int(bwdKMax)),
                                     threadsPerThreadgroup: tg)
        }
    }

    func encodeProjShBwdAdam(_ enc: MTLComputeCommandEncoder)
	{
        enc.setComputePipelineState(ctx.project_and_sh_backward)
        enc.setScalar(Int32(p.numPoints), 0)
		enc.setTensor(p.splatBuffers.means3d, 1)
		enc.setTensor(p.splatBuffers.scales, 2)
		enc.setScalar(p.globScale, 3)
		enc.setTensor(p.splatBuffers.quats, 4)
        // Pass matrices directly as bytes instead of MTensor buffers
        var viewmat = p.viewmat
        var projmat = p.projmat
        enc.setBytes(&viewmat, length: MemoryLayout<simd_float4x4>.size, index: 5)
        enc.setBytes(&projmat, length: MemoryLayout<simd_float4x4>.size, index: 6)
        enc.setArray(k.projIntrins, 7)
        enc.setArray(k.imgSize2, 8)
        enc.setTensor(c.radiiOut, 9)
		enc.setTensor(c.conics, 10)
        enc.setTensor(c.vXy, 11)
		enc.setTensor(c.vDepth, 12)
		enc.setTensor(c.vConic, 13)
        enc.setTensor(c.vMean3d, 14)
		enc.setTensor(c.vScale, 15)
		enc.setTensor(c.vQuat, 16)
        enc.setScalar(p.degree, 17)
		enc.setScalar(p.degreesToUse, 18)
        enc.setArray(k.camPos4, 19)
        enc.setTensor(c.vColorsRast, 20)
        // Fused SH backward + Adam: params + optimizer state instead of grad buffers
        enc.setTensor(t.adamParams[3], 21)    // features_dc params
        enc.setTensor(t.adamParams[4], 22)    // features_rest params
        enc.setTensor(t.adamExpAvg[3], 23)    // dc exp_avg
        enc.setTensor(t.adamExpAvgSq[3], 24)  // dc exp_avg_sq
        enc.setTensor(t.adamExpAvg[4], 25)    // rest exp_avg
        enc.setTensor(t.adamExpAvgSq[4], 26)  // rest exp_avg_sq
        enc.setScalar(shAdam, 27)
        enc.dispatch1D(p.numPoints, ctx.project_and_sh_backward)

        // Adam for remaining groups (3 = featuresDc, 4 = featuresRest fused above)
        guard numAdamGroups > 0 else { return }
        enc.barrier()
        for g in 0..<numAdamGroups where g != 3 && g != 4 {
            let n = UInt32(t.adamParams[g].numel)
            if n == 0 { continue }
            enc.setComputePipelineState(ctx.fused_adam)
            enc.setTensor(t.adamParams[g], 0)
            enc.setTensor(adamGrads[g], 1)
            enc.setTensor(t.adamExpAvg[g], 2)
            enc.setTensor(t.adamExpAvgSq[g], 3)
            enc.setScalar(t.adamStepSizes[g], 4)
            enc.setScalar(t.adamBeta1, 5)
            enc.setScalar(t.adamBeta2, 6)
            enc.setScalar(t.adamBc2Sqrts[g], 7)
            enc.setScalar(t.adamEps, 8)
            enc.setScalar(n, 9)
            enc.dispatch1D(Int(n), ctx.fused_adam)
        }
    }

    func encodeGradStats(_ enc: MTLComputeCommandEncoder) {
        enc.setComputePipelineState(ctx.accumulate_grad_stats)
        enc.setScalar(Int32(p.numPoints), 0)
        enc.setTensor(c.radiiOut, 1)
        enc.setTensor(c.vXy, 2)
        enc.setTensor(t.visCounts, 3)
        enc.setTensor(t.xysGradNorm, 4)
        enc.setTensor(t.max2dSize, 5)
        enc.setScalar(t.invMaxDim, 6)
        enc.dispatch1D(p.numPoints, ctx.accumulate_grad_stats)
    }

    func blitZero(_ cb: MTLCommandBuffer) {
        let blit = cb.makeBlitCommandEncoder()!
        blit.zero(c.lossSum)
        blit.zero(c.overflowFlag)
        blit.zero(c.tileScatterCounters)
        blit.zero(c.vXy)
        blit.zero(c.vConic)
        blit.zero(c.vColorsRast)
        blit.zero(c.vOpacity)
        blit.zero(c.vMean3d)
        blit.zero(c.vScale)
        blit.zero(c.vQuat)
        // v_features_dc / v_features_rest not needed — SH grads fused into Adam
        blit.endEncoding()
    }

    if !Profiling.profileStagesChecked {
        Profiling.profileStages = env("PROFILE_STAGES")
        Profiling.profileStagesChecked = true
    }

    // Execute training step and wait for completion
    try await ctx.executeAsync(jobName: "Train")
	{
		cb in
        if Profiling.profileStages && ctx.counterSamplingAvailable, let csb = ctx.counterSampleBuffer {
            // Per-stage profiling: separate encoders on the same CB with timestamp bookends.
            blitZero(cb)

            func profiledEncoder(_ stage: Int) -> MTLComputeCommandEncoder {
                let pd = MTLComputePassDescriptor()
                let att = pd.sampleBufferAttachments[0]!
                att.sampleBuffer = csb
                att.startOfEncoderSampleIndex = stage * 2
                att.endOfEncoderSampleIndex = stage * 2 + 1
                return cb.makeComputeCommandEncoder(descriptor: pd)!
            }

            let stages: [(MTLComputeCommandEncoder) -> Void] = [
                { encodeProjSh($0, pso, p, k, ctx.tensorCache) },
                { encodePrefixMap($0, pso, p, k, ctx.tensorCache) },
                { encodeRastFwd($0, pso, p, k, kMax: kMax, ctx.tensorCache) },
                encodeLossFwdBwd,
                encodeRastBwd,
                encodeProjShBwdAdam,
                encodeGradStats,
            ]
            for (i, encodeStage) in stages.enumerated() {
                let enc = profiledEncoder(i)
                encodeStage(enc)
                enc.endEncoding()
            }

            let ticksToMs = ctx.ticksToMs
            cb.addCompletedHandler { _ in
                autoreleasepool {
            
                    do
                    {
                        
                        let n = (nTrainStages - 1) * 2
                        guard let data = try csb.resolveCounterRange(0..<n) else { return }
                        data.withUnsafeBytes { raw in
                            let samples = raw.bindMemory(to: MTLCounterResultTimestamp.self)
                            Profiling.lock.lock(); defer { Profiling.lock.unlock() }
                            for i in 0..<(nTrainStages - 1) {
                                let start = samples[i * 2].timestamp
                                let end = samples[i * 2 + 1].timestamp
                                if start == UInt64.max || end == UInt64.max { continue }  // MTLCounterErrorValue
                                // encoder 0..6 → stage names 1..7 (blit_zero isn't sampled)
                                Profiling.stageTimes[i + 1].append(Double(end &- start) * ticksToMs)
                            }
                            Profiling.stageReportCount += 1
                            if Profiling.stageReportCount % 500 == 0 {
                                var report = "\n  === GPU Stage Profile (n=\(Profiling.stageReportCount)) ===\n"
                                var totalMedian = 0.0
                                for i in 1..<nTrainStages {
                                    let v = Profiling.stageTimes[i]
                                    if v.isEmpty { continue }
                                    let sorted = v.sorted()
                                    let med = sorted[sorted.count / 2]
                                    totalMedian += med
                                    report += "  " + pad20(trainStageNames[i])
                                    + String(format: " median=%.3fms  mean=%.3fms\n",
                                             med, sorted.reduce(0, +) / Double(sorted.count))
                                }
                                report += "  " + pad20("TOTAL (sum medians)") + String(format: " %.3fms", totalMedian)
                                logErr(report)
                            }
                        }
                    }
                    catch
                    {
                        logErr(error.localizedDescription)
                    }
                }
            }
        } else {
            // Production: single encoder for everything
            blitZero(cb)
            let enc = cb.makeComputeCommandEncoder()!
            encodeProjSh(enc, pso, p, k, ctx.tensorCache)
            enc.barrier()
            encodePrefixMap(enc, pso, p, k, ctx.tensorCache)
            enc.barrier()
            encodeRastFwd(enc, pso, p, k, kMax: kMax, ctx.tensorCache)
            enc.barrier()
            encodeLossFwdBwd(enc)
            enc.barrier()
            encodeRastBwd(enc)
            enc.barrier()
            encodeProjShBwdAdam(enc)
            enc.barrier()
            encodeGradStats(enc)
            enc.endEncoding()
        }
    }

    let loss = c.lossSum.floats()[0] / Float(h * w)
    return (c.radiiOut, loss)
}

private func pad20(_ s: String) -> String {
    s.count >= 20 ? s : s + String(repeating: " ", count: 20 - s.count)
}

// MARK: - GPU densification

public struct DensifyParams {
    var n: Int
    var bufCapacity: Int
    var gradThresh: Float, sizeThresh: Float, screenThresh: Float
    var checkScreen: Int32
    var cullAlphaThresh: Float, cullScaleThresh: Float, cullScreenSize: Float
    var checkHuge: Int32
    var xysGradNorm: MTensor, visCounts: MTensor, max2dSize: MTensor
    var halfMaxDim: Float
    var meansBuf: MTensor, scalesBuf: MTensor, quatsBuf: MTensor
    var featuresDcBuf: MTensor, featuresRestBuf: MTensor, opacitiesBuf: MTensor
    var frStride: Int
    var adamExpAvgBuf: [MTensor], adamExpAvgSqBuf: [MTensor]
    var splitFlag: MTensor, dupFlag: MTensor
    var splitPrefix: MTensor, dupPrefix: MTensor
    var keepFlag: MTensor, keepPrefix: MTensor
    var blockTotals: MTensor, compactScratch: MTensor
    var randomSamples: MTensor
}

/// Entire classify → grow → cull → compact pipeline in one encoder.
/// Returns the new number of active gaussians (one GPU→CPU sync).
func msplatDensify(_ d: DensifyParams, _ ctx: MetalContext) async throws -> Int {

    // Worst case: every gaussian splits (2 children) + dups (1 copy) = 3N
    let worstCase = 3 * d.n
    precondition(worstCase <= d.bufCapacity, "gpu_densify: 3*N exceeds buf_capacity")

    let logSizeFac = logf(1.6)

    // 6 params + 12 optimizer states, in a fixed order with their row strides
    let allBufs: [MTensor] = [
        d.meansBuf, d.scalesBuf, d.quatsBuf, d.featuresDcBuf, d.featuresRestBuf, d.opacitiesBuf,
    ] + d.adamExpAvgBuf + d.adamExpAvgSqBuf
    let s6 = [3, 3, 4, 3, d.frStride, 1]
    let allStrides = s6 + s6 + s6

    let nU32 = UInt32(d.n)
    let k = (d.n + 1023) / 1024  // threadgroups for prefix sum on N elements
    let frStride = Int32(d.frStride)

    // Execute densification and wait for completion
    try await ctx.executeAsync(jobName: "Denisfy")
	{
		cb in
        let enc = cb.makeComputeCommandEncoder()!

        func prefixSum(count: UInt32, groups: Int, flags: MTensor, prefix: MTensor) {
            let tg = MTLSize(width: 1024, height: 1, depth: 1)
            enc.setComputePipelineState(ctx.block_reduce)
            enc.setScalar(count, 0); enc.setTensor(flags, 1)
            enc.setTensor(d.blockTotals, 2)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.barrier()
            enc.setComputePipelineState(ctx.block_scan_propagate)
            enc.setScalar(count, 0); enc.setTensor(flags, 1)
            enc.setTensor(prefix, 2); enc.setTensor(d.blockTotals, 3)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.barrier()
        }

		// ---- Stage 1: classify (split / dup) ----
		enc.setComputePipelineState(ctx.densify_classify)
		enc.setScalar(nU32, 0)
		enc.setTensor(d.xysGradNorm, 1)
		enc.setTensor(d.visCounts, 2)
		enc.setTensor(d.scalesBuf, 3)
		enc.setTensor(d.max2dSize, 4)
		enc.setScalar(d.halfMaxDim, 5)
		enc.setScalar(d.gradThresh, 6)
		enc.setScalar(d.sizeThresh, 7)
		enc.setScalar(d.screenThresh, 8)
		enc.setScalar(d.checkScreen, 9)
		enc.setTensor(d.splitFlag, 10)
		enc.setTensor(d.dupFlag, 11)
		enc.dispatch1D(d.n, ctx.densify_classify)
		enc.barrier()

		// ---- Stages 2–3: prefix sums ----
		prefixSum(count: nU32, groups: k, flags: d.splitFlag, prefix: d.splitPrefix)
		prefixSum(count: nU32, groups: k, flags: d.dupFlag, prefix: d.dupPrefix)

		// ---- Stage 4: append split children ----
		enc.setComputePipelineState(ctx.densify_append_split)
		enc.setScalar(nU32, 0)
		enc.setTensor(d.splitFlag, 1)
		enc.setTensor(d.splitPrefix, 2)
		enc.setTensor(d.randomSamples, 3)
		enc.setScalar(logSizeFac, 4)
		enc.setTensor(d.meansBuf, 5)
		enc.setTensor(d.scalesBuf, 6)
		enc.setTensor(d.quatsBuf, 7)
		enc.setTensor(d.featuresDcBuf, 8)
		enc.setTensor(d.featuresRestBuf, 9)
		enc.setTensor(d.opacitiesBuf, 10)
		enc.setScalar(frStride, 11)
		for g in 0..<6 { enc.setTensor(d.adamExpAvgBuf[g], 12 + g) }
		for g in 0..<6 { enc.setTensor(d.adamExpAvgSqBuf[g], 18 + g) }
		enc.dispatch1D(d.n, ctx.densify_append_split)
		enc.barrier()

		// ---- Stage 5: append duplicates ----
		enc.setComputePipelineState(ctx.densify_append_dup)
		enc.setScalar(nU32, 0)
		enc.setTensor(d.dupFlag, 1)
		enc.setTensor(d.dupPrefix, 2)
		enc.setTensor(d.splitPrefix, 3)
		enc.setTensor(d.meansBuf, 4)
		enc.setTensor(d.scalesBuf, 5)
		enc.setTensor(d.quatsBuf, 6)
		enc.setTensor(d.featuresDcBuf, 7)
		enc.setTensor(d.featuresRestBuf, 8)
		enc.setTensor(d.opacitiesBuf, 9)
		enc.setScalar(frStride, 10)
		for g in 0..<6 { enc.setTensor(d.adamExpAvgBuf[g], 11 + g) }
		for g in 0..<6 { enc.setTensor(d.adamExpAvgSqBuf[g], 17 + g) }
		enc.dispatch1D(d.n, ctx.densify_append_dup)
		enc.barrier()

		// ---- Stage 6: cull classify on the grown population ----
		// Dispatch worst_case threads; the kernel derives N_new from the prefix sums.
		enc.setComputePipelineState(ctx.densify_cull_classify)
		enc.setScalar(nU32, 0)
		enc.setTensor(d.splitPrefix, 1)
		enc.setTensor(d.dupPrefix, 2)
		enc.setTensor(d.splitFlag, 3)
		enc.setTensor(d.opacitiesBuf, 4)
		enc.setTensor(d.scalesBuf, 5)
		enc.setTensor(d.max2dSize, 6)
		enc.setScalar(d.cullAlphaThresh, 7)
		enc.setScalar(d.cullScaleThresh, 8)
		enc.setScalar(d.cullScreenSize, 9)
		enc.setScalar(d.checkHuge, 10)
		enc.setScalar(d.checkScreen, 11)
		enc.setTensor(d.keepFlag, 12)
		enc.dispatch1D(worstCase, ctx.densify_cull_classify)
		enc.barrier()

		// ---- Stage 7: prefix sum on keep_flag over worst_case elements ----
		let wc = UInt32(worstCase)
		prefixSum(count: wc, groups: (worstCase + 1023) / 1024, flags: d.keepFlag, prefix: d.keepPrefix)

		// ---- Stage 8: compact each buffer via scratch, then copy back ----
		for b in 0..<18 {
			let strideU32 = UInt32(allStrides[b])
			let totalThreads = Int(wc * strideU32)

			enc.setComputePipelineState(ctx.compact_scatter)
			enc.setTensor(allBufs[b], 0)
			enc.setTensor(d.compactScratch, 1)
			enc.setTensor(d.keepPrefix, 2)
			enc.setTensor(d.keepFlag, 3)
			enc.setScalar(wc, 4)
			enc.setScalar(strideU32, 5)
			enc.dispatch1D(totalThreads, ctx.compact_scatter)
			enc.barrier()

			let lastIdx = wc - 1
			enc.setComputePipelineState(ctx.compact_copy_back)
			enc.setTensor(d.compactScratch, 0)
			enc.setTensor(allBufs[b], 1)
			enc.setTensor(d.keepPrefix, 2)
			enc.setScalar(lastIdx, 3)
			enc.setScalar(strideU32, 4)
			// C++ sizes this group's threadgroup from compact_scatter's limit too.
			let tpg = min(ctx.compact_scatter.maxTotalThreadsPerThreadgroup, totalThreads)
			enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
								threadsPerThreadgroup: MTLSize(width: max(tpg, 1), height: 1, depth: 1))
			enc.barrier()
		}

		enc.endEncoding()
	}

    // GPU work is complete - read new count from keep_prefix[worst_case - 1]
    return Int(d.keepPrefix.int32s()[worstCase - 1])
}
