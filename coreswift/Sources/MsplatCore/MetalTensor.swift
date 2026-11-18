import Foundation
import Metal
import simd


public enum DType: UInt8 {
    case float32, int32, int64, uint8, float64

    var size: Int {
        switch self {
        case .float32, .int32: return 4
        case .int64, .float64: return 8
        case .uint8: return 1
        }
    }
}

public struct MsplatError: LocalizedError
{
	public let errorDescription : String?

	public init(_ message: String)
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

public struct MTensor 
{
	var buffer: MTLBuffer?
	{
		didSet
		{
			OnStorageChanged()
		}
	}

	var cpuStorage: CPUStorage?
	{
		didSet
		{
			OnStorageChanged()
		}
	}
    public private(set) var shape: [Int] = []
	public private(set) var dtype: DType = .float32
	public private(set) var numel: Int = 0

    /// Undefined tensor (C++ default constructor).
    init() {}

    /// GPU allocation (shared storage). Throws if the device can't allocate.
    init(device: MTLDevice, shape: [Int], dtype: DType) throws 
	{
		if !shape.allSatisfy({ $0 > 0 })
		{
			throw MsplatError("tensor with a 0-width dimension; \(shape)")
		}
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

    public func size(_ dim: Int) -> Int {
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

    public func floats() -> UnsafeMutablePointer<Float> { pointer(Float.self) }
	public func int32s() -> UnsafeMutablePointer<Int32> { pointer(Int32.self) }
	public func float2s() -> UnsafeMutablePointer<simd_float2> { pointer(simd_float2.self) }
	public func float3s() -> UnsafeMutablePointer<simd_float3> { pointer(simd_float3.self) }
	public func float4s() -> UnsafeMutablePointer<simd_float4> { pointer(simd_float4.self) }
	public func quat4s() -> UnsafeMutablePointer<simd_quatf> { pointer(simd_quatf.self) }
	
	/// Get a buffer pointer with count for safer access
	public func floatBuffer() -> UnsafeMutableBufferPointer<Float> {
		guard dtype == .float32 else {
			fatalError("MTensor: floatBuffer() called on non-float32 tensor")
		}
		return UnsafeMutableBufferPointer(start: floats(), count: numel)
	}
	
	/// Copy tensor data to a Swift array
	public func toFloatArray() -> [Float] {
		guard dtype == .float32 else {
			fatalError("MTensor: toFloatArray() called on non-float32 tensor")
		}
		return Array(floatBuffer())
	}

    /// Convert 4x4 tensor to simd_float4x4 matrix (assumes row-major layout)
    func toFloat4x4() -> simd_float4x4 {
        precondition(shape.count == 2 && shape[0] == 4 && shape[1] == 4, "toFloat4x4: expected [4,4] shape")
        let f = floats()
        return simd_float4x4(
            SIMD4<Float>(f[0], f[1], f[2], f[3]),
            SIMD4<Float>(f[4], f[5], f[6], f[7]),
            SIMD4<Float>(f[8], f[9], f[10], f[11]),
            SIMD4<Float>(f[12], f[13], f[14], f[15])
        )
    }

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
	
	func OnStorageChanged()
	{
		if self.buffer != nil && self.cpuStorage != nil
		{
			print("Tensor with gpu & cpu data")
		}
	}
	
	static public func makeEmpty(device:MTLDevice,_ shape: [Int],dtype:DType = .float32) throws -> MTensor {
		return try MTensor(device: device, shape: shape, dtype: dtype)
	}
	
	/// Create a zero-initialized tensor with the given shape and data type.
	static public func makeZeros(device:MTLDevice,_ shape: [Int],dtype: DType = .float32) throws -> MTensor {
		let t = try makeEmpty(device:device, shape, dtype: dtype)
		t.zero()
		return t
	}
}
