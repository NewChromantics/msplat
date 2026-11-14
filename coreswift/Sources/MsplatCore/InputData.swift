// Port of input_data.hpp / input_data.cpp: Camera, Points, InputData, train/test
// split, auto scale & centre, camera export and the format dispatcher.

import Foundation

/// Image source for camera metadata
public enum ImageSource {
    case url(URL)
    case data(RGBImage)
}

/// Camera metadata: fixed data loaded from dataset (intrinsics, pose, dimensions).
/// Reference type to allow mutation of intrinsics during image loading.
public final class CameraMeta {
    var width = 0, height = 0
    var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
    var k1: Float = 0, k2: Float = 0, k3: Float = 0, p1: Float = 0, p2: Float = 0
    /// 4x4 row-major camera-to-world (OpenGL: Y-up, Z-back).
    var camToWorld = [Float](repeating: 0, count: 16)
    var imageSource: ImageSource
    
    init(imageSource: ImageSource = .data(RGBImage())) {
        self.imageSource = imageSource
    }
    
    /// URL accessor for image source
    var imageURL: URL? {
        get {
            if case .url(let url) = imageSource { return url }
            return nil
        }
        set {
            if let url = newValue {
                imageSource = .url(url)
            }
        }
    }

    var hasDistortion: Bool { k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0 }
}

/// Runtime camera data: lazily-built caches (image pyramids, GPU data, view matrices)
/// that are filled in during training. Reference type to allow mutation of caches.
public final class RuntimeCamera {
    let meta: CameraMeta
    private var _loadedImage: RGBImage?
    var imagePyramids: [Int: RGBImage] = [:]
    var gpuImageCache: [Int: MTensor] = [:]
    var cachedViewMat = MTensor(), cachedProjViewMat = MTensor()
    var cachedCamPos: (Float, Float, Float) = (0, 0, 0)
    var cachedFovX: Float = 0, cachedFovY: Float = 0

    init(meta: CameraMeta) {
        self.meta = meta
    }
    
    /// Load and process the base image with optional downscaling and distortion correction
    func loadImage(downscaleFactor: Float) throws {
        var raw: RGBImage
        
        switch meta.imageSource {
        case .url(let url):
            raw = try imreadRGB(url.path)
        case .data(let image):
            raw = image
        }

        // If actual image dimensions differ from metadata, rescale intrinsics
        if meta.width > 0 && meta.height > 0 && (raw.width != meta.width || raw.height != meta.height) {
            let sx = Float(raw.width) / Float(meta.width)
            let sy = Float(raw.height) / Float(meta.height)
            meta.fx *= sx; meta.fy *= sy; meta.cx *= sx; meta.cy *= sy
            meta.width = raw.width; meta.height = raw.height
        } else if meta.width == 0 || meta.height == 0 {
            meta.width = raw.width; meta.height = raw.height
        }

        if downscaleFactor > 1.0 {
            let newW = Int(Float(meta.width) / downscaleFactor)
            let newH = Int(Float(meta.height) / downscaleFactor)
            raw = resizeArea(raw, newW, newH)
            let s = 1.0 / downscaleFactor
            meta.fx *= s; meta.fy *= s; meta.cx *= s; meta.cy *= s
            meta.width = newW; meta.height = newH
        }

        if meta.hasDistortion {
            let r = undistortImage(raw, fx: meta.fx, fy: meta.fy, cx: meta.cx, cy: meta.cy,
                                   k1: meta.k1, k2: meta.k2, p1: meta.p1, p2: meta.p2, k3: meta.k3)
            raw = r.image
            meta.fx = r.fx; meta.fy = r.fy; meta.cx = r.cx; meta.cy = r.cy
            meta.width = r.width; meta.height = r.height
            meta.k1 = 0; meta.k2 = 0; meta.k3 = 0; meta.p1 = 0; meta.p2 = 0
        }

        _loadedImage = raw
        // Clear pyramids since base image changed
        imagePyramids.removeAll()
        gpuImageCache.removeAll()
    }
    
    /// Get the base loaded image
    var image: RGBImage {
        if let loaded = _loadedImage { return loaded }
        // Fallback: try to load from source
        if case .data(let img) = meta.imageSource { return img }
        return RGBImage()
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
    
    /// Downscale all camera images by the given factor
    /// - Parameter factor: Downscale factor (e.g., 2.0 means half the resolution)
    public func downscaleCameraImages(factor: Float) throws {
        guard factor > 0 else { return }
        for cam in cameras {
            try cam.loadImage(downscaleFactor: factor)
        }
    }
    
    /// Number of cameras in the dataset
    public var numCameras: Int { cameras.count }
    
    /// Camera-to-world pose (4x4 row-major, OpenGL) of a camera, or nil.
    public func cameraPose(at index: Int) -> [Float]? {
        cameras.indices.contains(index) ? cameras[index].meta.camToWorld : nil
    }

    func saveCameras(to url: URL, keepCrs: Bool) throws {
        var arr: [[String: Any]] = []
        for cam in cameras {
            let m = cam.meta.camToWorld
            // Undo OpenGL flip (negate columns 1, 2 back to OpenCV convention)
            let R: [Float] = [m[0], -m[1], -m[2], m[4], -m[5], -m[6], m[8], -m[9], -m[10]]
            var T: [Float] = [m[3], m[7], m[11]]
            if keepCrs {
                T[0] = T[0] / scale + translation.0
                T[1] = T[1] / scale + translation.1
                T[2] = T[2] / scale + translation.2
            }
            
            let fileName: String
            if let imageURL = cam.meta.imageURL {
                fileName = imageURL.lastPathComponent
            } else {
                fileName = ""
            }
            
            arr.append([
                "file_path": fileName,
                "width": cam.meta.width, "height": cam.meta.height,
                "fx": cam.meta.fx, "fy": cam.meta.fy, "cx": cam.meta.cx, "cy": cam.meta.cy,
                "rotation": [[R[0], R[1], R[2]], [R[3], R[4], R[5]], [R[6], R[7], R[8]]],
                "translation": T,
            ])
        }
        let data = try JSONSerialization.data(withJSONObject: arr, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
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
