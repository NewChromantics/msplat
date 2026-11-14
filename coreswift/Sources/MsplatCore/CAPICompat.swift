// Source-compatible stand-in for msplat_c_api.h.
//
// Lets existing Swift code written against the C API (e.g. swift/Sources/Msplat)
// build unchanged against this pure-Swift module. Handles are class references, so
// the *_destroy functions are no-ops: lifetime is managed by ARC.
//
// The C++ API lets exceptions escape through the C boundary (i.e. aborts); here
// failures in non-throwing entry points are fatal errors with the message. Prefer
// SplatDataset / SplatTrainer directly for throwing APIs.

import Foundation

public typealias MsplatDataset = SplatDataset
public typealias MsplatTrainer = SplatTrainer

/// RGB float32 pixels, HWC. `data` is malloc'd — caller must free() it.
public struct MsplatPixelBuffer {
    public var data: UnsafeMutablePointer<Float>?
    public var width: Int32
    public var height: Int32
}

public func msplat_default_config() -> MsplatConfig { MsplatConfig() }

private func fatal(_ what: String, _ error: Error) -> Never {
    fatalError("msplat: \(what): \(error)")
}

// MARK: Dataset

public func msplat_dataset_create(_ path: String, _ downscaleFactor: Float,
                                  _ evalMode: Bool, _ testEvery: Int32) -> MsplatDataset {
    do {
        return try SplatDataset(path: path, downscaleFactor: downscaleFactor,
                                evalMode: evalMode, testEvery: Int(testEvery))
    } catch { fatal("dataset_create", error) }
}

public func msplat_dataset_destroy(_ ds: MsplatDataset) {}

public func msplat_dataset_num_train(_ ds: MsplatDataset) -> Int32 { Int32(ds.numTrain) }
public func msplat_dataset_num_test(_ ds: MsplatDataset) -> Int32 { Int32(ds.numTest) }

/// Leaves `camToWorld` untouched for an out-of-range index (as the C++ does).
public func msplat_dataset_camera_pose(_ ds: MsplatDataset, _ cameraIndex: Int32,
                                       _ camToWorld: UnsafeMutablePointer<Float>) {
    guard let pose = ds.cameraPose(at: Int(cameraIndex)) else { return }
    for i in 0..<16 { camToWorld[i] = pose[i] }
}

// MARK: Trainer

public func msplat_trainer_create(_ ds: MsplatDataset, _ config: MsplatConfig) -> MsplatTrainer {
    do { return try SplatTrainer(dataset: ds, config: config) }
    catch { fatal("trainer_create", error) }
}

public func msplat_trainer_destroy(_ t: MsplatTrainer) {}

public func msplat_trainer_step(_ t: MsplatTrainer) -> MsplatStats { t.step() }
public func msplat_trainer_train(_ t: MsplatTrainer) { t.train() }
public func msplat_trainer_evaluate(_ t: MsplatTrainer) -> MsplatEvalMetrics { t.evaluate() }

private func toPixelBuffer(_ img: SplatImage?) -> MsplatPixelBuffer {
    guard let img, !img.pixels.isEmpty else { return MsplatPixelBuffer(data: nil, width: 0, height: 0) }
    let p = malloc(img.pixels.count * MemoryLayout<Float>.size)!.assumingMemoryBound(to: Float.self)
    img.pixels.withUnsafeBufferPointer { p.initialize(from: $0.baseAddress!, count: $0.count) }
    return MsplatPixelBuffer(data: p, width: Int32(img.width), height: Int32(img.height))
}

public func msplat_trainer_render(_ t: MsplatTrainer, _ cameraIndex: Int32, _ useTest: Bool) -> MsplatPixelBuffer {
    toPixelBuffer(t.render(cameraIndex: Int(cameraIndex), useTest: useTest))
}

public func msplat_trainer_render_pose(_ t: MsplatTrainer, _ camToWorld: UnsafePointer<Float>,
                                       _ refCameraIndex: Int32) -> MsplatPixelBuffer {
    let pose = Array(UnsafeBufferPointer(start: camToWorld, count: 16))
    return toPixelBuffer(t.renderFromPose(camToWorld: pose, refCameraIndex: Int(refCameraIndex)))
}

public func msplat_trainer_render_pose_to_buffer(_ t: MsplatTrainer, _ camToWorld: UnsafePointer<Float>,
                                                 _ refCameraIndex: Int32,
                                                 _ outRGBA: UnsafeMutablePointer<UInt8>?,
                                                 _ outWidth: UnsafeMutablePointer<Int32>,
                                                 _ outHeight: UnsafeMutablePointer<Int32>) {
    let pose = Array(UnsafeBufferPointer(start: camToWorld, count: 16))
    let (w, h) = t.renderFromPose(camToWorld: pose, refCameraIndex: Int(refCameraIndex), into: outRGBA)
    outWidth.pointee = Int32(w)
    outHeight.pointee = Int32(h)
}

public func msplat_trainer_export_ply(_ t: MsplatTrainer, _ path: String) {
    do { try t.exportPly(to: path) } catch { fatal("export_ply", error) }
}

public func msplat_trainer_export_splat(_ t: MsplatTrainer, _ path: String) {
    do { try t.exportSplat(to: path) } catch { fatal("export_splat", error) }
}

public func msplat_trainer_save_checkpoint(_ t: MsplatTrainer, _ path: String) {
    do { try t.saveCheckpoint(to: path) } catch { fatal("save_checkpoint", error) }
}

public func msplat_trainer_load_checkpoint(_ t: MsplatTrainer, _ path: String) -> Int32 {
    do { return Int32(try t.loadCheckpoint(from: path)) } catch { fatal("load_checkpoint", error) }
}

public func msplat_trainer_splat_count(_ t: MsplatTrainer) -> Int32 { Int32(t.splatCount) }
public func msplat_trainer_iteration(_ t: MsplatTrainer) -> Int32 { Int32(t.iteration) }

// MARK: Lifecycle

public func msplat_set_metallib_path(_ path: String?) { MsplatRuntime.setMetallibPath(path) }
public func msplat_sync() { MsplatRuntime.sync() }
public func msplat_cleanup() { MsplatRuntime.cleanup() }
