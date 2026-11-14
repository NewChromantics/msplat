// Port of input_data.hpp / input_data.cpp: Camera, Points, InputData, train/test
// split, auto scale & centre, camera export and the format dispatcher.

import Foundation

/// Image source for camera metadata
public enum ImageSource {
    case url(URL)
    case data(RGBImage)
}

/// Camera metadata: fixed data loaded from dataset (intrinsics, pose, dimensions).
/// Value type for camera parameters.
public struct CameraMeta 
{
    public var width = 0, height = 0
	public var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
	public var k1: Float = 0, k2: Float = 0, k3: Float = 0, p1: Float = 0, p2: Float = 0
    /// 4x4 row-major camera-to-world (OpenGL: Y-up, Z-back).
	public var camToWorld = [Float](repeating: 0, count: 16)
	public var imageSource: ImageSource
    
    public init(imageSource: ImageSource) {
        self.imageSource = imageSource
    }

    var hasDistortion: Bool { k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0 }
}

/// Runtime camera data: lazily-built caches (image pyramids, GPU data, view matrices)
/// that are filled in during training. Reference type to allow mutation of caches.
public final class RuntimeCamera 
{
    public let meta: CameraMeta
	public let image: RGBImage
    var imagePyramids: [Int: RGBImage] = [:]
    var gpuImageCache: [Int: MTensor] = [:]
    var cachedViewMat = MTensor(), cachedProjViewMat = MTensor()
    var cachedCamPos: (Float, Float, Float) = (0, 0, 0)
    var cachedFovX: Float = 0, cachedFovY: Float = 0

    public init(meta: CameraMeta, downscaleFactor: Float = 1.0) throws {
        // Load raw image from source
        var raw: RGBImage
        switch meta.imageSource {
        case .url(let url):
            raw = try RGBImage(contentsOf: url)
        case .data(let img):
            raw = img
        }
        
        // Make meta mutable for processing
        var processedMeta = meta
        
        // If actual image dimensions differ from metadata, rescale intrinsics
        if processedMeta.width > 0 && processedMeta.height > 0 && (raw.width != processedMeta.width || raw.height != processedMeta.height) {
            let sx = Float(raw.width) / Float(processedMeta.width)
            let sy = Float(raw.height) / Float(processedMeta.height)
            processedMeta.fx *= sx
            processedMeta.fy *= sy
            processedMeta.cx *= sx
            processedMeta.cy *= sy
            processedMeta.width = raw.width
            processedMeta.height = raw.height
        } else if processedMeta.width == 0 || processedMeta.height == 0 {
            processedMeta.width = raw.width
            processedMeta.height = raw.height
        }

        // Apply downscaling if requested
        if downscaleFactor > 1.0 {
            let newW = Int(Float(processedMeta.width) / downscaleFactor)
            let newH = Int(Float(processedMeta.height) / downscaleFactor)
            raw = resizeArea(raw, newW, newH)
            let s = 1.0 / downscaleFactor
            processedMeta.fx *= s
            processedMeta.fy *= s
            processedMeta.cx *= s
            processedMeta.cy *= s
            processedMeta.width = newW
            processedMeta.height = newH
        }

        // Undistort if needed
        if processedMeta.hasDistortion {
            let r = undistortImage(raw, fx: processedMeta.fx, fy: processedMeta.fy, 
                                   cx: processedMeta.cx, cy: processedMeta.cy,
                                   k1: processedMeta.k1, k2: processedMeta.k2, 
                                   p1: processedMeta.p1, p2: processedMeta.p2, 
                                   k3: processedMeta.k3)
            raw = r.image
            processedMeta.fx = r.fx
            processedMeta.fy = r.fy
            processedMeta.cx = r.cx
            processedMeta.cy = r.cy
            processedMeta.width = r.width
            processedMeta.height = r.height
            processedMeta.k1 = 0
            processedMeta.k2 = 0
            processedMeta.k3 = 0
            processedMeta.p1 = 0
            processedMeta.p2 = 0
        }
        
        self.meta = processedMeta
        self.image = raw
    }
    
    /// Create a RuntimeCamera with new metadata but reusing the existing image and caches.
    /// Used for operations like scaling and centering that only modify camera transforms.
    fileprivate init(meta: CameraMeta, image: RGBImage, copyingCachesFrom source: RuntimeCamera) {
        self.meta = meta
        self.image = image
        self.imagePyramids = source.imagePyramids
        self.gpuImageCache = source.gpuImageCache
        self.cachedViewMat = source.cachedViewMat
        self.cachedProjViewMat = source.cachedProjViewMat
        self.cachedCamPos = source.cachedCamPos
        self.cachedFovX = source.cachedFovX
        self.cachedFovY = source.cachedFovY
    }

    func getImage(downscaleFactor: Int) -> RGBImage {
        if downscaleFactor <= 1 { return image }
        if let img = imagePyramids[downscaleFactor] { return img }
        let scaled = resizeArea(image, image.width / downscaleFactor, image.height / downscaleFactor)
        imagePyramids[downscaleFactor] = scaled
        return scaled
    }

    func getGPUImage(downscaleFactor: Int) -> MTensor {
        if let t = gpuImageCache[downscaleFactor] { return t }
        let img = getImage(downscaleFactor: downscaleFactor)
        let mt = gpuEmpty([img.height, img.width, 3], .float32)
        mt.copyIn(img.data)
        gpuImageCache[downscaleFactor] = mt
        return mt
    }
}

struct Points {
    var xyz: [Float] = []   // N*3
    var rgb: [UInt8] = []   // N*3
    var count = 0
    
    init() {}
    
    init(xyz: [Float], rgb: [UInt8], count: Int) {
        self.xyz = xyz
        self.rgb = rgb
        self.count = count
    }
}

/// Lightweight dataset containing only camera metadata (no loaded images).
/// This is what loaders produce before images are loaded.
public final class InputData {
    public var cameras: [CameraMeta] = []
    var scale: Float = 1.0
    var translation: (Float, Float, Float) = (0, 0, 0)
    var points = Points()
    
    /// Number of cameras in the dataset
    public var numCameras: Int { cameras.count }
    
    public init() {}
}

/// Full dataset with loaded images and runtime caches for training.
/// Created from InputData by loading and processing camera images.
public final class SplatInputData {
    public var cameras: [RuntimeCamera] = []
    var scale: Float = 1.0
    var translation: (Float, Float, Float) = (0, 0, 0)
    var points = Points()
    
    /// Number of cameras in the dataset
    public var numCameras: Int { cameras.count }
    
    public init() {}
    
    /// Create SplatInputData from InputData by loading camera images.
    /// - Parameters:
    ///   - inputData: The metadata-only dataset
    ///   - downscaleFactor: Optional downscale factor to apply during loading (default: 1.0)
    public init(from inputData: InputData, downscaleFactor: Float = 1.0) throws {
        self.scale = inputData.scale
        self.translation = inputData.translation
        self.points = inputData.points
        
        // Load all camera images
        for meta in inputData.cameras {
            let runtimeCam = try RuntimeCamera(meta: meta, downscaleFactor: downscaleFactor)
            cameras.append(runtimeCam)
        }
    }
    
    /// Downscale all camera images by the given factor.
    /// This creates new RuntimeCamera instances with downscaled images.
    /// - Parameter factor: Downscale factor (e.g., 2.0 for half resolution)
    public func downscaleCameraImages(factor: Float) throws {
        guard factor > 1.0 else { return }
        
        var newCameras: [RuntimeCamera] = []
        for camera in cameras {
            // Create a new RuntimeCamera with the downscale applied
            let newCam = try RuntimeCamera(meta: camera.meta, downscaleFactor: factor)
            newCameras.append(newCam)
        }
        cameras = newCameras
    }
    
    /// Returns a new SplatInputData with cameras scaled and centered.
    /// The returned instance has camera positions normalized so that:
    /// - The mean camera position is at the origin
    /// - The maximum absolute coordinate is scaled to 1.0
    /// Point cloud data is also transformed accordingly.
    public func autoScaleAndCenter() -> SplatInputData {
        guard !cameras.isEmpty else { return self }
        
        // Calculate mean camera position
        var mean: (Float, Float, Float) = (0, 0, 0)
        for cam in cameras {
            mean.0 += cam.meta.camToWorld[3]
            mean.1 += cam.meta.camToWorld[7]
            mean.2 += cam.meta.camToWorld[11]
        }
        let n = Float(cameras.count)
        mean.0 /= n; mean.1 /= n; mean.2 /= n
        
        // Create new cameras with translated positions
        var translatedCameras: [RuntimeCamera] = []
        for cam in cameras {
            var newMeta = cam.meta
            newMeta.camToWorld[3] -= mean.0
            newMeta.camToWorld[7] -= mean.1
            newMeta.camToWorld[11] -= mean.2
            
            let newCam = RuntimeCamera(meta: newMeta, image: cam.image, copyingCachesFrom: cam)
            translatedCameras.append(newCam)
        }
        
        // Find maximum absolute coordinate
        var maxAbs: Float = 0
        for cam in translatedCameras {
            maxAbs = max(maxAbs, abs(cam.meta.camToWorld[3]))
            maxAbs = max(maxAbs, abs(cam.meta.camToWorld[7]))
            maxAbs = max(maxAbs, abs(cam.meta.camToWorld[11]))
        }
        let scale = maxAbs > 0 ? 1.0 / maxAbs : 1.0
        
        // Create final scaled cameras
        var scaledCameras: [RuntimeCamera] = []
        for cam in translatedCameras {
            var newMeta = cam.meta
            newMeta.camToWorld[3] *= scale
            newMeta.camToWorld[7] *= scale
            newMeta.camToWorld[11] *= scale
            
            let newCam = RuntimeCamera(meta: newMeta, image: cam.image, copyingCachesFrom: cam)
            scaledCameras.append(newCam)
        }
        
        // Transform point cloud
        var newPoints = Points(xyz: points.xyz, rgb: points.rgb, count: points.count)
        for i in 0..<newPoints.count {
            newPoints.xyz[i * 3 + 0] = (newPoints.xyz[i * 3 + 0] - mean.0) * scale
            newPoints.xyz[i * 3 + 1] = (newPoints.xyz[i * 3 + 1] - mean.1) * scale
            newPoints.xyz[i * 3 + 2] = (newPoints.xyz[i * 3 + 2] - mean.2) * scale
        }
        
        // Create new SplatInputData instance
        let result = SplatInputData()
        result.cameras = scaledCameras
        result.scale = scale
        result.translation = mean
        result.points = newPoints
        
        return result
    }
}

// MARK: - Scale & centre

func autoScaleAndCenter(_ data: InputData) {
    guard !data.cameras.isEmpty else { return }

    var mean: (Float, Float, Float) = (0, 0, 0)
    for cam in data.cameras {
        mean.0 += cam.camToWorld[3]
        mean.1 += cam.camToWorld[7]
        mean.2 += cam.camToWorld[11]
    }
    let n = Float(data.cameras.count)
    mean.0 /= n; mean.1 /= n; mean.2 /= n
    data.translation = mean

    for i in 0..<data.cameras.count {
        data.cameras[i].camToWorld[3] -= mean.0
        data.cameras[i].camToWorld[7] -= mean.1
        data.cameras[i].camToWorld[11] -= mean.2
    }

    var maxAbs: Float = 0
    for cam in data.cameras {
        maxAbs = max(maxAbs, abs(cam.camToWorld[3]))
        maxAbs = max(maxAbs, abs(cam.camToWorld[7]))
        maxAbs = max(maxAbs, abs(cam.camToWorld[11]))
    }
    data.scale = maxAbs > 0 ? 1.0 / maxAbs : 1.0

    for i in 0..<data.cameras.count {
        data.cameras[i].camToWorld[3] *= data.scale
        data.cameras[i].camToWorld[7] *= data.scale
        data.cameras[i].camToWorld[11] *= data.scale
    }

    let s = data.scale
    for i in 0..<data.points.count {
        data.points.xyz[i * 3 + 0] = (data.points.xyz[i * 3 + 0] - mean.0) * s
        data.points.xyz[i * 3 + 1] = (data.points.xyz[i * 3 + 1] - mean.1) * s
        data.points.xyz[i * 3 + 2] = (data.points.xyz[i * 3 + 2] - mean.2) * s
    }
}
