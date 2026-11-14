import Foundation
import NerfStudioData
import ColmapData
import PolycamData




public struct MsplatConfig {
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

/// Rendered RGB float32 image, HWC.
public struct SplatImage {
    public var pixels: [Float]
    public var width: Int
    public var height: Int
}

// MARK: - Runtime / lifecycle

public enum MsplatRuntime {
    /// Use a specific precompiled metallib (call before creating a trainer).
    public static func setMetallibPath(_ path: String?) { msplatSetMetallibPath(path) }
    /// Commit pending GPU work and wait for it.
    public static func sync() { msplatGpuSync() }
    /// Release cached GPU buffers.
    public static func cleanup() { cleanupMsplatMetal() }
    /// Capture to .gputrace (launch with METAL_CAPTURE_ENABLED=1). Sync on both edges.
    @discardableResult
    public static func beginGpuCapture(path: String) -> Bool { msplatBeginGpuCapture(path: path) }
    public static func endGpuCapture() { msplatEndGpuCapture() }
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
    private let model: Model
    private let config: MsplatConfig
    private let dataset: SplatInputData   // retained: no dangling dataset as in the C++
    private var currentStep = 0

    private var camIndices: [Int]
    private var camIterPos = 0
    private var rng = MT19937(seed: 42)

    public init(dataset: SplatInputData, config: MsplatConfig = MsplatConfig()) throws {
        _ = try MetalContext.shared()
        self.config = config
        self.dataset = dataset
        model = Model(
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
        shuffleLibcxx(&camIndices, &rng)
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
    public func step() -> MsplatStats {
        currentStep += 1
        let cam = dataset.cameras[nextCamera()]
        let gt = cam.getGPUImage(downscaleFactor: model.getDownscaleFactor(currentStep))

        let t0 = DispatchTime.now().uptimeNanoseconds
        model.fullIteration(cam, step: currentStep, gt: gt, ssimWeight: config.ssimWeight)
        model.schedulersStep(currentStep)
        model.afterTrain(currentStep)
        msplatCommit()
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
    public func train(callbackEvery: Int = 0, progress: ((MsplatStats) -> Void)? = nil) {
        while currentStep < Int(config.iterations) {
            let s = step()
            if let progress, callbackEvery > 0, currentStep % callbackEvery == 0 { progress(s) }
        }
    }

    /// Evaluate on held-out test cameras.
    public func evaluate() -> MsplatEvalMetrics {
        // No separate test cameras anymore - returns empty metrics
        return MsplatEvalMetrics()
    }

    private func renderToImage(_ cam: RuntimeCamera) -> SplatImage {
        let rgb = model.render(cam, step: currentStep)
        msplatGpuSync()
        let h = rgb.size(0), w = rgb.size(1)
        let pixels = Array(UnsafeBufferPointer(start: rgb.floats(), count: h * w * 3))
        return SplatImage(pixels: pixels, width: w, height: h)
    }

    /// Render a camera view. Nil if the index is out of range.
    public func render(cameraIndex: Int) -> SplatImage? {
        guard dataset.cameras.indices.contains(cameraIndex) else { return nil }
        return renderToImage(dataset.cameras[cameraIndex])
    }

    /// Camera at `refCameraIndex` with its pose replaced (intrinsics kept).
    private func poseCamera(_ camToWorld: [Float], _ refCameraIndex: Int) throws -> RuntimeCamera? {
        precondition(camToWorld.count == 16, "camToWorld must have 16 elements")
        guard dataset.cameras.indices.contains(refCameraIndex) else { return nil }
        let refCam = dataset.cameras[refCameraIndex]
        
        // Create new meta with same intrinsics but different pose
        var newMeta = refCam.meta
        newMeta.camToWorld = camToWorld
        
        // Create runtime camera with new meta
        let cam = try RuntimeCamera(meta: newMeta)
        // Share image pyramids and GPU cache from reference camera
        cam.imagePyramids = refCam.imagePyramids
        cam.gpuImageCache = refCam.gpuImageCache
        // Leave cached matrices empty - they'll be recomputed from new pose
        
        return cam
    }

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention),
    /// using the intrinsics of a reference training camera.
    public func renderFromPose(camToWorld: [Float], refCameraIndex: Int = 0) throws -> SplatImage? {
        guard let cam = try poseCamera(camToWorld, refCameraIndex) else { return nil }
        return renderToImage(cam)
    }

    /// Render from pose straight into a caller-provided RGBA8 buffer (≥ w*h*4 bytes).
    /// Pass nil to just query the output size. Returns (0, 0) for a bad camera index.
    @discardableResult
    public func renderFromPose(camToWorld: [Float], refCameraIndex: Int = 0,
                               into rgba: UnsafeMutablePointer<UInt8>?) throws -> (width: Int, height: Int) {
        guard let cam = try poseCamera(camToWorld, refCameraIndex) else { return (0, 0) }
        let rgb = model.render(cam, step: currentStep)
        msplatGpuSync()
        let h = rgb.size(0), w = rgb.size(1)
        guard let out = rgba else { return (w, h) }
        // Unified memory: read the GPU tensor directly
        let src = rgb.floats()
        for i in 0..<(w * h) {
            out[i * 4 + 0] = unitToByte(src[i * 3 + 0])
            out[i * 4 + 1] = unitToByte(src[i * 3 + 1])
            out[i * 4 + 2] = unitToByte(src[i * 3 + 2])
            out[i * 4 + 3] = 255
        }
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
}

/// (uint8_t)(fminf(fmaxf(v, 0), 1) * 255) — NaN maps to 0 like fmaxf.
@inline(__always)
func unitToByte(_ v: Float) -> UInt8 {
    UInt8(Float.minimum(Float.maximum(v, 0), 1) * 255)
}
