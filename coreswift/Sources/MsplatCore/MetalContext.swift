// Port of the context / lifecycle half of metal/msplat_metal.mm:
// device + queue, pipeline states, command-buffer lifecycle (MPSCommandBuffer with
// commitAndContinue), metallib discovery, GPU capture and timing/profiling hooks.

import Foundation
import Metal
import MetalPerformanceShaders
import PopCommon


struct GpuExecuteStats
{
	var name : String
	var gpuExecutionDurationMs : Int
	var gpuScheduleDurationMs : Int
}

// MARK: - Profiling state (PROFILE_GPU / PROFILE_STAGES)

let trainStageNames = [
    "blit_zero", "proj_sh_fwd", "prefix_sort_pack", "rast_fwd",
    "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam", "grad_stats",
]
let nTrainStages = 8

final class Profiling {
    nonisolated(unsafe) static var gpuTimingEnabled = false
    nonisolated(unsafe) static var gpuTimingChecked = false
    nonisolated(unsafe) static var profileStages = false
    nonisolated(unsafe) static var profileStagesChecked = false

    static let lock = NSLock()
    nonisolated(unsafe) static var gpuTimesMs: [Double] = []
    nonisolated(unsafe) static var stageTimes = [[Double]](repeating: [], count: nTrainStages)
    nonisolated(unsafe) static var stageReportCount = 0
}

func env(_ name: String) -> Bool { ProcessInfo.processInfo.environment[name] != nil }

func logErr(_ s: String) {
    FileHandle.standardError.write((s.hasSuffix("\n") ? s : s + "\n").data(using: .utf8)!)
}


//	metal objects are threadsafe, stats are not 
public final class MetalContext : @unchecked Sendable 
{
    let device: MTLDevice
    let queue: MTLCommandQueue

    // Per-stage GPU timestamp profiling
    var counterSampleBuffer: MTLCounterSampleBuffer?
    var counterSamplingAvailable = false
    var ticksToMs = 0.0

    // Forward pipeline
    let project_and_sh_forward: MTLComputePipelineState
    let nd_rasterize_forward: MTLComputePipelineState
    // Tile-local sorting
    let scatter_to_prealloc_bins: MTLComputePipelineState
    let bitonic_sort_per_tile: MTLComputePipelineState
    // Prefix sum
    let prefix_sum: MTLComputePipelineState
    let block_reduce: MTLComputePipelineState
    let block_scan_propagate: MTLComputePipelineState
    // Depth-chunked rasterization
    let rasterize_forward_chunked: MTLComputePipelineState
    let rasterize_forward_merge: MTLComputePipelineState
    let compute_chunk_prefix_suffix: MTLComputePipelineState
    let rasterize_backward_chunked: MTLComputePipelineState
    let rasterize_backward: MTLComputePipelineState
    // Separable SSIM loss
    let ssim_h_fwd: MTLComputePipelineState
    let ssim_v_fwd: MTLComputePipelineState
    let ssim_fused_v_fwd_h_bwd: MTLComputePipelineState
    let ssim_v_bwd: MTLComputePipelineState
    // Backward
    let project_and_sh_backward: MTLComputePipelineState
    let fused_adam: MTLComputePipelineState
    let accumulate_grad_stats: MTLComputePipelineState
    // GPU densification
    let densify_classify: MTLComputePipelineState
    let densify_append_split: MTLComputePipelineState
    let densify_append_dup: MTLComputePipelineState
    let densify_cull_classify: MTLComputePipelineState
    let compact_scatter: MTLComputePipelineState
    let compact_copy_back: MTLComputePipelineState

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw MsplatError("msplat: no Metal device available")
        }
        device = dev
        guard let q = dev.makeCommandQueue() else {
            throw MsplatError("msplat: failed to create command queue")
        }
        queue = q

        let library = try MetalContext.loadLibrary(device: dev)

        func load(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw MsplatError("msplat: kernel not found: \(name)")
            }
            do {
                return try dev.makeComputePipelineState(function: fn)
            } catch {
                throw MsplatError("msplat: failed to create pipeline for \(name): \(error)")
            }
        }

        project_and_sh_forward      = try load("project_and_sh_forward_kernel")
        nd_rasterize_forward        = try load("nd_rasterize_forward_kernel")
        scatter_to_prealloc_bins    = try load("scatter_to_prealloc_bins_kernel")
        bitonic_sort_per_tile       = try load("bitonic_sort_per_tile_kernel")
        prefix_sum                  = try load("prefix_sum_kernel")
        block_reduce                = try load("block_reduce_kernel")
        block_scan_propagate        = try load("block_scan_propagate_kernel")
        rasterize_forward_chunked   = try load("rasterize_forward_chunked_kernel")
        rasterize_forward_merge     = try load("rasterize_forward_merge_kernel")
        compute_chunk_prefix_suffix = try load("compute_chunk_prefix_suffix_kernel")
        rasterize_backward_chunked  = try load("rasterize_backward_chunked_kernel")
        rasterize_backward          = try load("rasterize_backward_kernel")
        ssim_h_fwd                  = try load("ssim_h_fwd_kernel")
        ssim_v_fwd                  = try load("ssim_v_fwd_kernel")
        ssim_fused_v_fwd_h_bwd      = try load("ssim_fused_v_fwd_h_bwd_kernel")
        ssim_v_bwd                  = try load("ssim_v_bwd_kernel")
        project_and_sh_backward     = try load("project_and_sh_backward_kernel")
        fused_adam                  = try load("fused_adam_kernel")
        accumulate_grad_stats       = try load("accumulate_grad_stats_kernel")
        densify_classify            = try load("densify_classify_kernel")
        densify_append_split        = try load("densify_append_split_kernel")
        densify_append_dup          = try load("densify_append_dup_kernel")
        densify_cull_classify       = try load("densify_cull_classify_kernel")
        compact_scatter             = try load("compact_scatter_kernel")
        compact_copy_back           = try load("compact_copy_back_kernel")

        if env("PROFILE_STAGES") {
            Profiling.profileStages = true
            Profiling.profileStagesChecked = true
            initCounterSampling()
        }
    }

    /// Metallib discovery order:
    ///  1. explicit path from msplatSetMetallibPath()
    ///  2. default.metallib in this module's resource bundle
    ///  3. default.metallib next to the main executable (CLI builds)
    ///  4. compile the bundled msplat_metal.metal source at runtime
    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary 
	{
        
        let fm = FileManager.default
        if let url = Bundle.module.url(forResource: "default", withExtension: "metallib"),
           let lib = try? device.makeLibrary(URL: url) {
            return lib
        }
        if let exe = Bundle.main.executableURL {
            let url = exe.deletingLastPathComponent().appendingPathComponent("default.metallib")
            if fm.fileExists(atPath: url.path), let lib = try? device.makeLibrary(URL: url) {
                return lib
            }
        }
        if let src = Bundle.module.url(forResource: "msplat_metal", withExtension: "metal") {
            do {
                let source = try String(contentsOf: src, encoding: .utf8)
                let opts = MTLCompileOptions()
                opts.mathMode = .fast  // matches `xcrun metal -O3` default fast-math
                return try device.makeLibrary(source: source, options: opts)
            } catch {
                throw MsplatError("msplat: failed to compile msplat_metal.metal: \(error)")
            }
        }
        throw MsplatError("msplat: failed to load metallib: default.metallib not found")
    }

    private func initCounterSampling() {
        let sampleCount = nTrainStages * 2
        guard let timestampSet = device.counterSets?.first(where: {
            $0.name == MTLCommonCounterSet.timestamp.rawValue
        }) else {
            logErr("PROFILE_STAGES: MTLCommonCounterSetTimestamp not available")
            return
        }
        guard device.supportsCounterSampling(.atStageBoundary) else {
            logErr("PROFILE_STAGES: AtStageBoundary sampling not supported")
            return
        }
        let desc = MTLCounterSampleBufferDescriptor()
        desc.counterSet = timestampSet
        desc.sampleCount = sampleCount
        desc.storageMode = .shared
        desc.label = "msplat stage profiling"
        do {
            counterSampleBuffer = try device.makeCounterSampleBuffer(descriptor: desc)
        } catch {
            logErr("PROFILE_STAGES: Failed to create counter sample buffer: \(error.localizedDescription)")
            return
        }
        // MTLCounterResultTimestamp is in nanoseconds on Apple Silicon.
        ticksToMs = 1.0 / 1e6
        counterSamplingAvailable = true
        logErr("PROFILE_STAGES: GPU timestamp profiling enabled (\(sampleCount) sample slots)")
    }

    // MARK: Command buffer lifecycle

    /// Create a fresh command buffer from this context's queue.
    func makeCommandBuffer() -> MTLCommandBuffer {
        return MPSCommandBuffer(from: queue)
    }
    
    /// Create a command buffer and add GPU timing handler if enabled.
    func makeCommandBufferWithTiming() -> MTLCommandBuffer {
        let cb = MPSCommandBuffer(from: queue)
        if Profiling.gpuTimingEnabled {
            cb.addCompletedHandler { cb in
                let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
                if ms > 0 {
                    Profiling.lock.lock()
                    Profiling.gpuTimesMs.append(ms)
                    Profiling.lock.unlock()
                }
            }
        }
        return cb
    }
    
    /// Execute GPU work asynchronously and wait for completion.
    /// Pass a closure that encodes work into the command buffer.
    @discardableResult
	func executeAsync(jobName:String,_ work: (MTLCommandBuffer) throws -> Void) async throws -> GpuExecuteStats 
	{
		let cb = makeCommandBufferWithTiming()
		try work(cb)
		
		let promise = SendablePromise<GpuExecuteStats>()
		cb.addCompletedHandler 
		{
			completionMeta in
			let executeMs = (completionMeta.gpuEndTime - completionMeta.gpuStartTime) * 1000.0
			let scheduleMs = (completionMeta.kernelEndTime - completionMeta.kernelStartTime) * 1000.0
			let meta = GpuExecuteStats(name: jobName, gpuExecutionDurationMs:Int(executeMs), gpuScheduleDurationMs: Int(scheduleMs) )
			promise.Resolve(meta)
		}
		cb.commit()
		let stats = try await promise.value
		print("\(jobName) took \(stats.gpuExecutionDurationMs)ms (\(stats.gpuScheduleDurationMs)ms setup)")
		return stats
	}

    // MARK: - Tensor creation
    
    /// Create an empty (uninitialized) tensor with the given shape and data type.
    public func makeEmpty(_ shape: [Int], _ dtype: DType = .float32) throws -> MTensor {
        return try MTensor(device: device, shape: shape, dtype: dtype)
    }

    /// Create a zero-initialized tensor with the given shape and data type.
    public func makeZeros(_ shape: [Int], _ dtype: DType = .float32) throws -> MTensor {
        let t = try makeEmpty(shape, dtype)
        t.zero()
        return t
    }
    
    // MARK: - GPU capture
    
    /// Capture GPU work to a .gputrace document. Needs METAL_CAPTURE_ENABLED=1 at launch.
    @discardableResult
    public func beginGpuCapture(path: String) -> Bool {
        let mgr = MTLCaptureManager.shared()
        guard mgr.supportsDestination(.gpuTraceDocument) else {
            logErr("msplat: .gputrace capture unsupported — launch with METAL_CAPTURE_ENABLED=1")
            return false
        }
        // startCapture fails outright if the destination already exists.
        try? FileManager.default.removeItem(atPath: path)
        let desc = MTLCaptureDescriptor()
        desc.captureObject = queue
        desc.destination = .gpuTraceDocument
        desc.outputURL = URL(fileURLWithPath: path)
        do {
            try mgr.startCapture(with: desc)
            return true
        } catch {
            logErr("msplat: startCapture failed: \(error.localizedDescription)")
            return false
        }
    }
    
    public func endGpuCapture() {
        MTLCaptureManager.shared().stopCapture()
    }
}

// MARK: - Free functions for profiling and timing

func msplatEnableGpuTiming(_ enable: Bool) {
    Profiling.gpuTimingEnabled = enable
    Profiling.gpuTimingChecked = true
}

/// Drains accumulated per-command-buffer GPU times (ms). Thread-safe.
func msplatDrainGpuTimes() -> [Double] {
    Profiling.lock.lock(); defer { Profiling.lock.unlock() }
    let out = Profiling.gpuTimesMs
    Profiling.gpuTimesMs.removeAll()
    return out
}

/// Drains per-stage GPU times, keyed by stage name, in pipeline order.
func msplatDrainStageTimes() -> [(name: String, timesMs: [Double])] {
    Profiling.lock.lock(); defer { Profiling.lock.unlock() }
    var out: [(String, [Double])] = []
    for i in 0..<nTrainStages {
        out.append((trainStageNames[i], Profiling.stageTimes[i]))
        Profiling.stageTimes[i].removeAll()
    }
    return out
}
