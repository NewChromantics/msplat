import Foundation

// MARK: - Shared Binary Reading Utilities

public struct BinaryReaderError: LocalizedError 
{
	public var errorDescription: String?
	
	public init(_ description: String) 
	{
		self.errorDescription = description
	}
}

/// Little-endian binary reader over Data.
/// Shared utility used by ColmapDataSource for reading COLMAP binary files.
struct BinaryReader {
	let data: Data
	var offset = 0
	
	init(_ data: Data) { self.data = data }
	
	var remaining: Int { data.count - offset }
	
	mutating func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
		let n = MemoryLayout<T>.size
		guard remaining >= n else { throw BinaryReaderError("Unexpected end of file") }
		let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
		offset += n
		return T(littleEndian: v)
	}
	
	mutating func readF64() throws -> Double { Double(bitPattern: try read(UInt64.self)) }
	mutating func readF32() throws -> Float { Float(bitPattern: try read(UInt32.self)) }
	
	mutating func readCString() throws -> String {
		var bytes: [UInt8] = []
		while remaining > 0 {
			let b = try read(UInt8.self)
			if b == 0 { break }
			bytes.append(b)
		}
		return String(decoding: bytes, as: UTF8.self)
	}
	
	mutating func skip(_ n: Int) throws {
		guard n >= 0, remaining >= n else { throw BinaryReaderError("Unexpected end of file") }
		offset += n
	}
}
