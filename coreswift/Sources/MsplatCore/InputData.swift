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
public struct CameraMeta {
    var width = 0, height = 0
    var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
    var k1: Float = 0, k2: Float = 0, k3: Float = 0, p1: Float = 0, p2: Float = 0
    /// 4x4 row-major camera-to-world (OpenGL: Y-up, Z-back).
    var camToWorld = [Float](repeating: 0, count: 16)
    var imageSource: ImageSource
    
    public init(imageSource: ImageSource) {
        self.imageSource = imageSource
    }

    var hasDistortion: Bool { k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0 }
}

/// Runtime camera data: lazily-built caches (image pyramids, GPU data, view matrices)
/// that are filled in during training. Reference type to allow mutation of caches.
public final class RuntimeCamera {
    var meta: CameraMeta
    let image: RGBImage
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
}

public final class InputData {
    public var cameras: [RuntimeCamera] = []
    var scale: Float = 1.0
    var translation: (Float, Float, Float) = (0, 0, 0)
    var points = Points()
    
    /// Number of cameras in the dataset
    public var numCameras: Int { cameras.count }
}

// MARK: - Scale & centre

func autoScaleAndCenter(_ data: InputData) {
    guard !data.cameras.isEmpty else { return }

    var mean: (Float, Float, Float) = (0, 0, 0)
    for cam in data.cameras {
        mean.0 += cam.meta.camToWorld[3]
        mean.1 += cam.meta.camToWorld[7]
        mean.2 += cam.meta.camToWorld[11]
    }
    let n = Float(data.cameras.count)
    mean.0 /= n; mean.1 /= n; mean.2 /= n
    data.translation = mean

    for cam in data.cameras {
        cam.meta.camToWorld[3] -= mean.0
        cam.meta.camToWorld[7] -= mean.1
        cam.meta.camToWorld[11] -= mean.2
    }

    var maxAbs: Float = 0
    for cam in data.cameras {
        maxAbs = max(maxAbs, abs(cam.meta.camToWorld[3]))
        maxAbs = max(maxAbs, abs(cam.meta.camToWorld[7]))
        maxAbs = max(maxAbs, abs(cam.meta.camToWorld[11]))
    }
    data.scale = maxAbs > 0 ? 1.0 / maxAbs : 1.0

    for cam in data.cameras {
        cam.meta.camToWorld[3] *= data.scale
        cam.meta.camToWorld[7] *= data.scale
        cam.meta.camToWorld[11] *= data.scale
    }

    let s = data.scale
    for i in 0..<data.points.count {
        data.points.xyz[i * 3 + 0] = (data.points.xyz[i * 3 + 0] - mean.0) * s
        data.points.xyz[i * 3 + 1] = (data.points.xyz[i * 3 + 1] - mean.1) * s
        data.points.xyz[i * 3 + 2] = (data.points.xyz[i * 3 + 2] - mean.2) * s
    }
}
