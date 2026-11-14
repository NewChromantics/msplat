// Port of include/metal_tensor.hpp
//
// MTensor is a lightweight tensor: shape metadata over either a shared-storage
// MTLBuffer (GPU, CPU-visible on Apple Silicon) or a plain CPU allocation.
//
// Semantics vs. the C++ class:
//  * Copies share storage (the C++ copy constructor CFRetains the buffer); same here
//    via ARC on the buffer / storage object.
//  * view(n) in C++ is NON-owning and can dangle. Here a view retains the buffer, so
//    it can't outlive its storage. Behaviour is otherwise identical.

import Foundation
import Metal

enum DType: UInt8 {
    case float32, int32, int64, uint8, float64

    var size: Int {
        switch self {
        case .float32, .int32: return 4
        case .int64, .float64: return 8
        case .uint8: return 1
        }
    }
}

struct MsplatError: LocalizedError
{
	let errorDescription : String?

	init(_ message: String)
	{
		errorDescription = message 
	}
}

/// Backing store for CPU-only tensors.
final class CPUStorage {
    let pointer: UnsafeMutableRawPointer
    let byteCount: Int
    init(byteCount: Int) {
        self.byteCount = byteCount
        pointer = .allocate(byteCount: max(byteCount, 1), alignment: 16)
        pointer.initializeMemory(as: UInt8.self, repeating: 0, count: max(byteCount, 1))
    }
    deinit { pointer.deallocate() }
}

struct MTensor {
    private(set) var buffer: MTLBuffer?
    private var cpuStorage: CPUStorage?
    private(set) var shape: [Int] = []
    private(set) var dtype: DType = .float32
    private(set) var numel: Int = 0

    /// Undefined tensor (C++ default constructor).
    init() {}

    /// GPU allocation (shared storage). Throws if the device can't allocate.
    init(device: MTLDevice, shape: [Int], dtype: DType) throws {
        self.shape = shape
        self.dtype = dtype
        numel = shape.reduce(1, *)
        var bytes = numel * dtype.size
        if bytes == 0 { bytes = 4 }
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MsplatError("msplat: MTLBuffer allocation failed for \(bytes) bytes; "
                + "device already holds \(device.currentAllocatedSize)")
        }
        buffer = buf
    }

    /// CPU allocation (no Metal buffer).
    init(cpuShape shape: [Int], dtype: DType) {
        self.shape = shape
        self.dtype = dtype
        numel = shape.reduce(1, *)
        cpuStorage = CPUStorage(byteCount: numel * dtype.size)
    }

    var defined: Bool { buffer != nil || cpuStorage != nil }
    var isGpu: Bool { buffer != nil }
    var ndim: Int { shape.count }
    var elementSize: Int { dtype.size }
    var nbytes: Int { numel * dtype.size }

    func size(_ dim: Int) -> Int {
        shape[dim < 0 ? dim + shape.count : dim]
    }

    /// Elements per row along dim 0.
    var stride0: Int {
        shape.count <= 1 ? 1 : shape.dropFirst().reduce(1, *)
    }

    var rawPointer: UnsafeMutableRawPointer {
        if let b = buffer { return b.contents() }
        if let c = cpuStorage { return c.pointer }
        fatalError("MTensor: access to undefined tensor")
    }

    func pointer<T>(_: T.Type = T.self) -> UnsafeMutablePointer<T> {
        rawPointer.assumingMemoryBound(to: T.self)
    }

    func floats() -> UnsafeMutablePointer<Float> { pointer(Float.self) }
    func int32s() -> UnsafeMutablePointer<Int32> { pointer(Int32.self) }

    func zero() {
        guard defined else { return }
        memset(rawPointer, 0, nbytes)
    }

    /// CPU copy of the data.
    func cpu() -> MTensor {
        let out = MTensor(cpuShape: shape, dtype: dtype)
        memcpy(out.rawPointer, rawPointer, nbytes)
        return out
    }

    mutating func reset() {
        buffer = nil
        cpuStorage = nil
        shape = []
        numel = 0
    }

    /// View of the first `n` rows along dim 0, sharing storage.
    func view(_ n: Int) -> MTensor {
        var v = MTensor()
        v.buffer = buffer
        v.cpuStorage = cpuStorage
        v.shape = shape
        v.shape[0] = n
        v.dtype = dtype
        v.numel = n * stride0
        return v
    }

    /// Copy `byteCount` bytes from `src` into this tensor.
    func copyIn(_ src: UnsafeRawPointer, byteCount: Int) {
        memcpy(rawPointer, src, byteCount)
    }

    func copyIn<T>(_ array: [T]) {
        guard !array.isEmpty else { return }
        array.withUnsafeBytes { copyIn($0.baseAddress!, byteCount: $0.count) }
    }
}
