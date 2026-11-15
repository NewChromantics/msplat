import Foundation
import NerfStudioData
import ColmapData
import PolycamData
import Accelerate
import simd


//	can we use simd_half3?
struct PackedHalf3 
{
	var x: Float16
	var y: Float16
	var z: Float16
}

//	simd_half4
struct PackedRGBHalf4 {
	var r: Float16
	var g: Float16
	var b: Float16
	var a: Float16
}



public struct PlySplatPoint : Sendable, Equatable
{
	public var position : simd_float3
	public var quaternionWxyz : simd_float4
	public var rotation : simd_quatf		{	simd_normalize( simd_quatf(ix: quaternionWxyz.y, iy: quaternionWxyz.z, iz: quaternionWxyz.w, r: quaternionWxyz.x) )	}
	public var scaleLog : simd_float3
	public var scale : simd_float3			{	simd_float3( exp(scaleLog.x), exp(scaleLog.y), exp(scaleLog.z) )	}
	public var opacityLogit : Float
	public var opacity : Float				{	1 / (1 + exp(-opacityLogit))	}
	public var shDc : simd_float3
	public var colour : simd_float3		{	simd_clamp( 0.5 + 0.28209479 * shDc, simd_float3(repeating: 0), simd_float3(repeating: 1) )	}
}


public struct MsplatConfig 
{
    public var iterations: Int32 = 30_000
    public var shDegree: Int32 = 3
    public var shDegreeInterval: Int32 = 1_000
    public var ssimWeight: Float = 0.2
    public var numDownscales: Int32 = 2
    public var resolutionSchedule: Int32 = 3_000
    public var refineEvery: Int32 = 100
    public var warmupLength: Int32 = 500
    public var resetAlphaEvery: Int32 = 30
    public var densifyGradThresh: Float = 0.0002
    public var densifySizeThresh: Float = 0.01
    public var stopScreenSizeAt: Int32 = 4_000
    public var splitScreenSize: Float = 0.05
    public var keepCrs: Bool = false
    /// Default magenta — high contrast, makes under-reconstructed regions obvious.
    public var bgColor: (Float, Float, Float) = (0.6130, 0.0101, 0.3984)

    public init() {}
}

public struct MsplatStats {
    public var iteration: Int32 = 0
    public var splatCount: Int32 = 0
    public var msPerStep: Float = 0
    public init() {}
}

public struct MsplatEvalMetrics {
    public var psnr: Float = 0
    public var ssim: Float = 0
    public var l1: Float = 0
    public var numTest: Int32 = 0
    public var numGaussians: Int32 = 0
    public init() {}
}


// MARK: - Runtime / lifecycle

public enum MsplatRuntime {
    public static func cleanup() { cleanupMsplatMetal() }
    /// Capture to .gputrace (launch with METAL_CAPTURE_ENABLED=1). Sync on both edges.
    //@discardableResult
    public static func enableGpuTiming(_ enable: Bool) { msplatEnableGpuTiming(enable) }
    public static func drainGpuTimes() -> [Double] { msplatDrainGpuTimes() }
    public static func drainStageTimes() -> [(name: String, timesMs: [Double])] { msplatDrainStageTimes() }
}



/// Protocol for all dataset source types (Nerfstudio, COLMAP, Polycam, etc.)
public protocol SplatDatasetSource 
{
	static func Detect(rootPath:URL) -> Bool
	
	func Load() async throws -> InputData
}

/// Factory for creating the appropriate dataset source based on the dataset format
public enum SplatDatasetLoader {
    /// Errors thrown during dataset detection or loading
    public enum LoaderError: Error, CustomStringConvertible {
        case pathDoesNotExist(String)
        case unrecognizedFormat(String)
        
        public var description: String {
            switch self {
            case .pathDoesNotExist(let path):
                return "Path does not exist: \(path)"
            case .unrecognizedFormat(let path):
                return "Unrecognized dataset format in: \(path)\n" +
                       "Supported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)"
            }
        }
    }
    
    /// Auto-detect the dataset format and create the appropriate source
    /// - Parameters:
    ///   - url: Path to the dataset directory
    ///   - colmapImagePath: Optional custom image path for COLMAP datasets
    /// - Returns: A SplatDatasetSource ready to load the dataset
    public static func createSource(
        from url: URL,
        colmapImagePath:URL?
    ) throws -> SplatDatasetSource 
	{
        let fm = FileManager.default
        
        guard fm.fileExists(atPath: url.path) else {
            throw LoaderError.pathDoesNotExist(url.path)
        }
        
        // Try each dataset source using their Detect methods
        if NerfstudioDatasetSource.Detect(rootPath: url) 
		{
			let nerf = NerfstudioDatasetSource(url: url)
            return nerf
        }
        
        if ColmapDatasetSource.Detect(rootPath: url) 
		{
            return ColmapDatasetSource(url: url, imageSourcePath: colmapImagePath)
        }
        
		if PolycamDatasetSource.Detect(rootPath: url)
		{
            return PolycamDatasetSource(url: url)
        }
        
        throw LoaderError.unrecognizedFormat(url.path)
    }
    
    /// Convenience method to auto-detect format and load the dataset in one step
    public static func load(
        from url: URL,
        colmapImagePath:URL?
    ) async throws -> InputData {
        let source = try createSource(from: url, colmapImagePath: colmapImagePath)
        return try await source.Load()
    }
}

// MARK: - Trainer

public final class SplatTrainer {
    private let context: MetalContext
    private let model: Model
    private let config: MsplatConfig
    private let dataset: SplatInputData   // retained: no dangling dataset as in the C++
    private var currentStep = 0

    private var camIndices: [Int]
    private var camIterPos = 0
	private var rng = MT19937(seed: 42)
		//private var rng = Xoshiro256PlusPlus(seed: 42)

    public init(context: MetalContext, dataset: SplatInputData, config: MsplatConfig = MsplatConfig()) throws {
        self.context = context
        self.config = config
        self.dataset = dataset
        model = try Model(
            metalContext: context,
            inputData: dataset, numCameras: dataset.cameras.count,
            numDownscales: Int(config.numDownscales), resolutionSchedule: Int(config.resolutionSchedule),
            shDegree: Int(config.shDegree), shDegreeInterval: Int(config.shDegreeInterval),
            refineEvery: Int(config.refineEvery), warmupLength: Int(config.warmupLength),
            resetAlphaEvery: Int(config.resetAlphaEvery),
            densifyGradThresh: config.densifyGradThresh, densifySizeThresh: config.densifySizeThresh,
            stopScreenSizeAt: Int(config.stopScreenSizeAt), splitScreenSize: config.splitScreenSize,
            maxSteps: Int(config.iterations), keepCrs: config.keepCrs,
            bgColor: config.bgColor)
        camIndices = Array(0..<dataset.cameras.count)
        shuffleCameras()
    }

    private func shuffleCameras() {
        shuffle(&camIndices, &rng)
        camIterPos = 0
    }

    private func nextCamera() -> Int {
        if camIterPos >= camIndices.count { shuffleCameras() }
        let i = camIndices[camIterPos]
        camIterPos += 1
        return i
    }

    /// Run one training step.
    @discardableResult
    public func step() async throws -> MsplatStats 
	{
        currentStep += 1
        let cam = dataset.cameras[nextCamera()]
        let gt = try cam.getGPUImage(context: context, downscaleFactor: model.getDownscaleFactor(currentStep))

        let t0 = DispatchTime.now().uptimeNanoseconds
        
        // Execute training iteration and wait for GPU completion
        try await model.fullIteration(cam, step: currentStep, gt: gt, ssimWeight: config.ssimWeight)
        model.schedulersStep(currentStep)
       try await  model.afterTrain(currentStep)
        
        let t1 = DispatchTime.now().uptimeNanoseconds

        var s = MsplatStats()
        s.iteration = Int32(currentStep)
        s.splatCount = Int32(model.means.size(0))
        // C++ truncates to whole microseconds before dividing.
        s.msPerStep = Float((t1 - t0) / 1000) / 1000.0
        return s
    }

    /// Train from the current iteration up to `config.iterations`.
    /// `progress` (optional) is called every `callbackEvery` steps.
    public func train(callbackEvery: Int = 0, progress: ((MsplatStats) -> Void)? = nil) async throws {
        while currentStep < Int(config.iterations) {
            let s = try await step()
            if let progress, callbackEvery > 0, currentStep % callbackEvery == 0 { progress(s) }
        }
    }

    /// Evaluate on held-out test cameras.
    public func evaluate() -> MsplatEvalMetrics {
        // No separate test cameras anymore - returns empty metrics
        return MsplatEvalMetrics()
    }

	private func renderToImage(_ cam: RuntimeCamera, format: RGBImage.ImageFormat) async throws -> RGBImage 
	{
        // Render executes its own GPU work and waits for completion
        let rgb = try await model.render(cam, step: currentStep, channels: format.channels)
        
        let h = rgb.size(0)
		let w = rgb.size(1)
		let pixels = Array(UnsafeBufferPointer(start: rgb.floats(), count: h * w * format.channels))
        return try RGBImage(data: pixels, width: w, height: h, format: format)
    }

    /// Render a camera view. Nil if the index is out of range.
    public func render(cameraIndex: Int, format: RGBImage.ImageFormat) async throws -> RGBImage
	{
		guard dataset.cameras.indices.contains(cameraIndex) else 
		{
			throw MsplatError("No such camera \(cameraIndex)/\(dataset.cameras.count)")
		}
		return try await renderToImage(dataset.cameras[cameraIndex], format: format)
    }
	
	public func GetSplats() -> [PlySplatPoint]
	{
		return model.GetSplats()
	}

    /// Camera at `refCameraIndex` with its pose replaced (intrinsics kept).
    private func GetNewRuntimeCamera(_ camToWorld: [Float], _ refCameraIndex: Int) throws -> RuntimeCamera
	{
        precondition(camToWorld.count == 16, "camToWorld must have 16 elements")
        guard dataset.cameras.indices.contains(refCameraIndex) else 
		{
			throw MsplatError("No such camera \(refCameraIndex)/\(dataset.cameras.count)")
		}
        let refCam = dataset.cameras[refCameraIndex]
        
        // Create new meta with same intrinsics but different pose
        var newMeta = refCam.meta
        newMeta.camToWorld = camToWorld
        
        // Create runtime camera with new meta
		let cam = try RuntimeCamera(meta: newMeta,copyImage:refCam.image)
        // Share image pyramids and GPU cache from reference camera
        cam.imagePyramids = refCam.imagePyramids
        cam.gpuImageCache = refCam.gpuImageCache
        // Leave cached matrices empty - they'll be recomputed from new pose
        
        return cam
    }

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention),
    /// using the intrinsics of a reference training camera.
    public func renderFromPose(camToWorld: [Float], refCameraIndex: Int = 0, format: RGBImage.ImageFormat) async throws -> RGBImage? 
	{
        let cam = try GetNewRuntimeCamera(camToWorld, refCameraIndex)
		return try await renderToImage(cam, format: format)
    }

    /// Render from pose straight into a caller-provided RGBA8 buffer (≥ w*h*4 bytes).
    /// Pass nil to just query the output size. Returns (0, 0) for a bad camera index.
    @discardableResult
    public func renderFromPose(camToWorld: [Float], refCameraIndex: Int = 0,
                               into rgba: UnsafeMutablePointer<UInt8>?) async throws -> (width: Int, height: Int)
	{
        let cam = try GetNewRuntimeCamera(camToWorld, refCameraIndex)
        
        // Render executes its own GPU work and waits for completion
        let rgb = try await model.render(cam, step: currentStep, channels: 4)
        
        let h = rgb.size(0), w = rgb.size(1)
        guard let out = rgba else { return (w, h) }
        
        // Convert RGB float to RGBA8
        convertRGBFloatToRGBA8_Accelerate(rgb.floats(), out, w * h)
        
        return (w, h)
    }

    public func exportPly(to path: String) throws { try model.savePly(path, step: currentStep) }
    public func exportSplat(to path: String) throws { try model.saveSplat(path) }

    /// Save full training state (params + optimizer) for resume.
    public func saveCheckpoint(to path: String) throws {
        try model.saveCheckpoint(path, step: currentStep)
    }

    /// Load a checkpoint and resume from its iteration (returned).
    @discardableResult
    public func loadCheckpoint(from path: String) throws -> Int {
        currentStep = try model.loadCheckpoint(path)
        shuffleCameras()
        return currentStep
    }

    public var splatCount: Int { model.means.size(0) }
    public var iteration: Int { currentStep }
    
    // MARK: - GPU Memory Tracking
    
    /// Get detailed GPU memory usage breakdown from the model
    public func getGPUMemoryBreakdown() -> GPUMemoryBreakdown {
        model.getGPUMemoryBreakdown()
    }
    
    /// Get total GPU memory usage in bytes (model + dataset caches)
    public var totalGPUMemoryUsage: Int {
        model.gpuMemoryUsage + Int(dataset.totalGPUMemoryUsage)
    }
    
    /// Get total GPU memory usage in megabytes
    public var totalGPUMemoryUsageMB: Double {
        Double(totalGPUMemoryUsage) / (1024.0 * 1024.0)
    }
    
    /// Get model-only GPU memory usage in bytes
    public var modelGPUMemoryUsage: Int {
        model.gpuMemoryUsage
    }
    
    /// Get model-only GPU memory usage in megabytes
    public var modelGPUMemoryUsageMB: Double {
        model.gpuMemoryUsageMB
    }
    
    /// Get dataset cache GPU memory usage in megabytes
    public var datasetGPUMemoryUsageMB: Double {
        dataset.totalGPUMemoryUsageMB
    }
    
    /// Get dataset CPU image memory usage in megabytes
    public var datasetCPUMemoryUsageMB: Double {
        dataset.totalCPUImageMemoryMB
    }
    
    /// Get memory efficiency (percentage of allocated capacity actually used)
    public var memoryEfficiency: Double {
        model.memoryEfficiency
    }
    
    /// Print comprehensive GPU memory report to console
    public func printGPUMemoryReport() {
        print("═══════════════════════════════════════════════════════")
        print("           GPU MEMORY REPORT")
        print("═══════════════════════════════════════════════════════")
        print("")
        model.printGPUMemoryReport()
        print("")
        print("Dataset Caches:       \(String(format: "%.2f MB", datasetGPUMemoryUsageMB))")
        print("───────────────────────────────────────────────────────")
        print("Grand Total:          \(String(format: "%.2f MB", totalGPUMemoryUsageMB))")
        print("═══════════════════════════════════════════════════════")
    }
}

/// (uint8_t)(fminf(fmaxf(v, 0), 1) * 255) — NaN maps to 0 like fmaxf.
@inline(__always)
func unitToByte(_ v: Float) -> UInt8 {
    UInt8(Float.minimum(Float.maximum(v, 0), 1) * 255)
}

/// Convert RGB float32 tensor to RGBA8 buffer.
/// - Parameters:
///   - src: Source RGB float data (normalized 0-1), HWC layout
///   - dest: Destination RGBA8 buffer (must be at least pixelCount * 4 bytes)
///   - pixelCount: Number of pixels to convert
@inline(__always)
func convertRGBFloatToRGBA8(_ src: UnsafePointer<Float>, 
                            _ dest: UnsafeMutablePointer<UInt8>, 
                            _ pixelCount: Int) {
    // Process in chunks for better cache performance
    let chunkSize = 1024
    for chunkStart in stride(from: 0, to: pixelCount, by: chunkSize) {
        let chunkEnd = min(chunkStart + chunkSize, pixelCount)
        
        for i in chunkStart..<chunkEnd {
            let srcIdx = i * 3
            let dstIdx = i * 4
            
            // Load RGB values
            let r = src[srcIdx]
            let g = src[srcIdx + 1]
            let b = src[srcIdx + 2]
            
            // Clamp and convert to bytes (compiler can vectorize this)
            dest[dstIdx]     = UInt8(min(max(r, 0), 1) * 255)
            dest[dstIdx + 1] = UInt8(min(max(g, 0), 1) * 255)
            dest[dstIdx + 2] = UInt8(min(max(b, 0), 1) * 255)
            dest[dstIdx + 3] = 255
        }
    }
}

// Alternative implementations for benchmarking:

/// Accelerate-based vectorized conversion (fastest for large images).
/// Requires: import Accelerate

@inline(__always)
func convertRGBFloatToRGBA8_Accelerate(_ src: UnsafePointer<Float>,
                                       _ dest: UnsafeMutablePointer<UInt8>,
                                       _ pixelCount: Int) {
    var tempR = [Float](repeating: 0, count: pixelCount)
    var tempG = [Float](repeating: 0, count: pixelCount)
    var tempB = [Float](repeating: 0, count: pixelCount)
    
    // Deinterleave RGB
    for i in 0..<pixelCount {
        tempR[i] = src[i * 3]
        tempG[i] = src[i * 3 + 1]
        tempB[i] = src[i * 3 + 2]
    }
    
    // Scale and clamp using Accelerate
    var scale: Float = 255.0
    var zero: Float = 0.0
    var twoFiftyFive: Float = 255.0
    
    vDSP_vsmul(tempR, 1, &scale, &tempR, 1, vDSP_Length(pixelCount))
    vDSP_vsmul(tempG, 1, &scale, &tempG, 1, vDSP_Length(pixelCount))
    vDSP_vsmul(tempB, 1, &scale, &tempB, 1, vDSP_Length(pixelCount))
    
    vDSP_vclip(tempR, 1, &zero, &twoFiftyFive, &tempR, 1, vDSP_Length(pixelCount))
    vDSP_vclip(tempG, 1, &zero, &twoFiftyFive, &tempG, 1, vDSP_Length(pixelCount))
    vDSP_vclip(tempB, 1, &zero, &twoFiftyFive, &tempB, 1, vDSP_Length(pixelCount))
    
    // Convert to UInt8 and interleave RGBA
    for i in 0..<pixelCount {
        dest[i * 4]     = UInt8(tempR[i])
        dest[i * 4 + 1] = UInt8(tempG[i])
        dest[i * 4 + 2] = UInt8(tempB[i])
        dest[i * 4 + 3] = 255
    }
}


/// Simple unrolled version (good baseline for small images).

@inline(__always)
func convertRGBFloatToRGBA8_Simple(_ src: UnsafePointer<Float>,
                                   _ dest: UnsafeMutablePointer<UInt8>,
                                   _ pixelCount: Int) {
    for i in 0..<pixelCount {
        dest[i * 4]     = unitToByte(src[i * 3])
        dest[i * 4 + 1] = unitToByte(src[i * 3 + 1])
        dest[i * 4 + 2] = unitToByte(src[i * 3 + 2])
        dest[i * 4 + 3] = 255
    }
}

