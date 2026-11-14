// Port of input_data.hpp / input_data.cpp: Camera, Points, InputData, train/test
// split, auto scale & centre, camera export and the format dispatcher.

import Foundation

/// A camera view. Reference type because it owns lazily-built caches (image pyramid,
/// GPU images, view matrices) that are filled in during training. The C++ copies
/// Camera by value in a few places; `copy()` provides that where it matters.
final class Camera {
    var width = 0, height = 0
    var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
    var k1: Float = 0, k2: Float = 0, k3: Float = 0, p1: Float = 0, p2: Float = 0
    /// 4x4 row-major camera-to-world (OpenGL: Y-up, Z-back).
    var camToWorld = [Float](repeating: 0, count: 16)
    var filePath = ""

    var image = RGBImage()
    var imagePyramids: [Int: RGBImage] = [:]
    var gpuImageCache: [Int: MTensor] = [:]
    var cachedViewMat = MTensor(), cachedProjViewMat = MTensor()
    var cachedCamPos: (Float, Float, Float) = (0, 0, 0)
    var cachedFovX: Float = 0, cachedFovY: Float = 0

    init() {}

    /// Value copy (C++ copy constructor semantics: caches are copied, GPU tensors shared).
    func copy() -> Camera {
        let c = Camera()
        c.width = width; c.height = height
        c.fx = fx; c.fy = fy; c.cx = cx; c.cy = cy
        c.k1 = k1; c.k2 = k2; c.k3 = k3; c.p1 = p1; c.p2 = p2
        c.camToWorld = camToWorld
        c.filePath = filePath
        c.image = image
        c.imagePyramids = imagePyramids
        c.gpuImageCache = gpuImageCache
        c.cachedViewMat = cachedViewMat; c.cachedProjViewMat = cachedProjViewMat
        c.cachedCamPos = cachedCamPos
        c.cachedFovX = cachedFovX; c.cachedFovY = cachedFovY
        return c
    }

    var hasDistortion: Bool { k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0 }

    func loadImage(downscaleFactor: Float) throws {
        var raw = try imreadRGB(filePath)
        if raw.isEmpty { return }

        // If actual image dimensions differ from metadata, rescale intrinsics
        if width > 0 && height > 0 && (raw.width != width || raw.height != height) {
            let sx = Float(raw.width) / Float(width)
            let sy = Float(raw.height) / Float(height)
            fx *= sx; fy *= sy; cx *= sx; cy *= sy
            width = raw.width; height = raw.height
        } else if width == 0 || height == 0 {
            width = raw.width; height = raw.height
        }

        if downscaleFactor > 1.0 {
            let newW = Int(Float(width) / downscaleFactor)
            let newH = Int(Float(height) / downscaleFactor)
            raw = resizeArea(raw, newW, newH)
            let s = 1.0 / downscaleFactor
            fx *= s; fy *= s; cx *= s; cy *= s
            width = newW; height = newH
        }

        if hasDistortion {
            let r = undistortImage(raw, fx: fx, fy: fy, cx: cx, cy: cy,
                                   k1: k1, k2: k2, p1: p1, p2: p2, k3: k3)
            raw = r.image
            fx = r.fx; fy = r.fy; cx = r.cx; cy = r.cy
            width = r.width; height = r.height
            k1 = 0; k2 = 0; k3 = 0; p1 = 0; p2 = 0
        }

        image = raw
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

final class InputData {
    var cameras: [Camera] = []
    var scale: Float = 1.0
    var translation: (Float, Float, Float) = (0, 0, 0)
    var points = Points()

    /// Returns (train cameras, optional validation camera).
    func getCameras(validate: Bool, valImage: String = "random") -> ([Camera], Camera?) {
        if !validate { return (cameras.map { $0.copy() }, nil) }

        var valIdx = -1
        if valImage == "random" {
            var rng = MT19937(seed: 42)
            valIdx = Int(rng.next() % UInt32(cameras.count))
        } else {
            valIdx = cameras.firstIndex { $0.filePath.contains(valImage) } ?? -1
        }
        if valIdx < 0 { valIdx = 0 }

        var train: [Camera] = []
        for (i, c) in cameras.enumerated() where i != valIdx { train.append(c.copy()) }
        return (train, cameras[valIdx])
    }

    /// Every `testEvery`-th camera (starting at 0) goes to test.
    func splitTrainTest(testEvery: Int) -> (train: [Camera], test: [Camera]) {
        var train: [Camera] = [], test: [Camera] = []
        for (i, c) in cameras.enumerated() {
            if i % testEvery == 0 { test.append(c.copy()) } else { train.append(c.copy()) }
        }
        return (train, test)
    }

    func saveCameras(to filename: String, keepCrs: Bool) throws {
        var arr: [[String: Any]] = []
        for cam in cameras {
            let m = cam.camToWorld
            // Undo OpenGL flip (negate columns 1, 2 back to OpenCV convention)
            let R: [Float] = [m[0], -m[1], -m[2], m[4], -m[5], -m[6], m[8], -m[9], -m[10]]
            var T: [Float] = [m[3], m[7], m[11]]
            if keepCrs {
                T[0] = T[0] / scale + translation.0
                T[1] = T[1] / scale + translation.1
                T[2] = T[2] / scale + translation.2
            }
            arr.append([
                "file_path": (cam.filePath as NSString).lastPathComponent,
                "width": cam.width, "height": cam.height,
                "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                "rotation": [[R[0], R[1], R[2]], [R[3], R[4], R[5]], [R[6], R[7], R[8]]],
                "translation": T,
            ])
        }
        let data = try JSONSerialization.data(withJSONObject: arr, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: filename))
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

    for cam in data.cameras {
        cam.camToWorld[3] -= mean.0
        cam.camToWorld[7] -= mean.1
        cam.camToWorld[11] -= mean.2
    }

    var maxAbs: Float = 0
    for cam in data.cameras {
        maxAbs = max(maxAbs, abs(cam.camToWorld[3]))
        maxAbs = max(maxAbs, abs(cam.camToWorld[7]))
        maxAbs = max(maxAbs, abs(cam.camToWorld[11]))
    }
    data.scale = maxAbs > 0 ? 1.0 / maxAbs : 1.0

    for cam in data.cameras {
        cam.camToWorld[3] *= data.scale
        cam.camToWorld[7] *= data.scale
        cam.camToWorld[11] *= data.scale
    }

    let s = data.scale
    for i in 0..<data.points.count {
        data.points.xyz[i * 3 + 0] = (data.points.xyz[i * 3 + 0] - mean.0) * s
        data.points.xyz[i * 3 + 1] = (data.points.xyz[i * 3 + 1] - mean.1) * s
        data.points.xyz[i * 3 + 2] = (data.points.xyz[i * 3 + 2] - mean.2) * s
    }
}

// MARK: - Format dispatcher

/// Auto-detect format (Nerfstudio, COLMAP, Polycam) and load.
func inputDataFromX(_ path: String, colmapImagePath: String = "") throws -> InputData {
    let fm = FileManager.default
    let root = URL(fileURLWithPath: path)
    func exists(_ parts: String...) -> Bool {
        fm.fileExists(atPath: parts.reduce(root) { $0.appendingPathComponent($1) }.path)
    }

    guard fm.fileExists(atPath: path) else { throw MsplatError("Path does not exist; \(path)") }

    if exists("transforms.json") { return try loadNerfstudio(path) }
    if exists("cameras.bin") || exists("sparse", "0", "cameras.bin") {
        return try loadColmap(path, imageSourcePath: colmapImagePath)
    }
    if exists("keyframes", "corrected_cameras") || exists("cameras.json") {
        return try loadPolycam(path)
    }
    throw MsplatError("Unrecognized dataset format in: \(path)\n"
        + "Supported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)")
}
