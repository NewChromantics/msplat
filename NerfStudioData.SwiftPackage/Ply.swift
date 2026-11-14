import Foundation



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


@inline(__always)
func clampToUInt8(_ v: Float) -> UInt8 {
	if v.isNaN { return 0 }
	return UInt8(min(max(v, 0), 255))
}


public struct PlyError: LocalizedError 
{
	public var errorDescription: String?
	
	public init(_ description: String) 
	{
		self.errorDescription = description
	}
}


public struct BinaryReaderError: LocalizedError 
{
	public var errorDescription: String?
	
	public init(_ description: String) 
	{
		self.errorDescription = description
	}
}



/// Little-endian binary reader over Data.
struct BinaryReader {
	let data: Data
	var offset = 0
	
	init(_ data: Data) { self.data = data }
	
	var remaining: Int { data.count - offset }
	
	mutating func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
		let n = MemoryLayout<T>.size
		guard remaining >= n else { throw PlyError("Unexpected end of file") }
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


func readPly(_ path: String) throws -> Points {
	guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
		throw PlyError("Cannot open PLY file: \(path)")
	}
	guard let header = plyHeader(data), let first = header.lines.first, first.contains("ply") else {
		throw PlyError("Not a PLY file: \(path)")
	}
	let lines = header.lines, bodyOffset = header.bodyOffset
	
	var binary = false, ascii = false
	var numVertices = 0
	var props: [PlyProp] = []
	var vertexBytes = 0
	var inVertexElement = false
	
	for line in lines.dropFirst() where line != "end_header" {
		let tok = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
		guard let head = tok.first else { continue }
		switch head {
			case "format":
				let fmt = tok.count > 1 ? tok[1] : ""
				binary = fmt == "binary_little_endian"
				ascii = fmt == "ascii"
			case "element":
				inVertexElement = tok.count > 1 && tok[1] == "vertex"
				if inVertexElement, tok.count > 2 { numVertices = Int(tok[2]) ?? 0 }
			case "property":
				// Only vertex properties. (The C++ also collected properties of any element
				// declared after "vertex", which corrupts the vertex stride if one exists.)
				if inVertexElement && numVertices > 0 && tok.count > 2 {
					let t = PlyType(tok[1])
					props.append(PlyProp(name: tok[2], type: t, offset: vertexBytes))
					vertexBytes += t.size
				}
			default:
				break
		}
	}
	
	if numVertices == 0 { throw PlyError("PLY has no vertices") }
	if !binary && !ascii { throw PlyError("Unsupported PLY format (need ascii or binary_little_endian)") }
	
	func find(_ n: String) -> Int? { props.firstIndex { $0.name == n } }
	guard let ix = find("x"), let iy = find("y"), let iz = find("z") else {
		throw PlyError("PLY missing x/y/z properties")
	}
	let ir = find("red"), ig = find("green"), ib = find("blue")
	let hasColor = ir != nil && ig != nil && ib != nil
	
	var pts = Points()
	pts.count = numVertices
	pts.xyz = [Float](repeating: 0, count: numVertices * 3)
	pts.rgb = [UInt8](repeating: 128, count: numVertices * 3) // default grey
	
	if binary {
		guard data.count - bodyOffset >= numVertices * vertexBytes else {
			throw PlyError("PLY truncated: \(path)")
		}
		data.withUnsafeBytes { raw in
			let base = raw.baseAddress!.advanced(by: bodyOffset)
			for i in 0..<numVertices {
				let rec = base.advanced(by: i * vertexBytes)
				func readFloat(_ pi: Int) -> Float {
					let p = props[pi]
					switch p.type {
						case .float32: return Float(bitPattern: UInt32(littleEndian: rec.loadUnaligned(fromByteOffset: p.offset, as: UInt32.self)))
						case .float64: return Float(Double(bitPattern: UInt64(littleEndian: rec.loadUnaligned(fromByteOffset: p.offset, as: UInt64.self))))
						default: return 0
					}
				}
				func readU8(_ pi: Int) -> UInt8 {
					let p = props[pi]
					switch p.type {
						case .uint8: return rec.load(fromByteOffset: p.offset, as: UInt8.self)
						case .uint16: return UInt8(UInt16(littleEndian: rec.loadUnaligned(fromByteOffset: p.offset, as: UInt16.self)) >> 8)
						case .float32: return clampToUInt8(readFloat(pi) * 255.0)
						default: return 128
					}
				}
				pts.xyz[i * 3 + 0] = readFloat(ix)
				pts.xyz[i * 3 + 1] = readFloat(iy)
				pts.xyz[i * 3 + 2] = readFloat(iz)
				if hasColor {
					pts.rgb[i * 3 + 0] = readU8(ir!)
					pts.rgb[i * 3 + 1] = readU8(ig!)
					pts.rgb[i * 3 + 2] = readU8(ib!)
				}
			}
		}
	} else {
		let body = String(decoding: data[(data.startIndex + bodyOffset)...], as: UTF8.self)
		var lineIter = body.split(separator: "\n", omittingEmptySubsequences: false).makeIterator()
		for i in 0..<numVertices {
			guard let line = lineIter.next() else { throw PlyError("PLY truncated: \(path)") }
			let tokens = line.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\r" })
			func f(_ idx: Int) throws -> Float {
				guard idx < tokens.count, let v = Float(tokens[idx]) else {
					throw PlyError("PLY parse error at vertex \(i): \(path)")
				}
				return v
			}
			pts.xyz[i * 3 + 0] = try f(ix)
			pts.xyz[i * 3 + 1] = try f(iy)
			pts.xyz[i * 3 + 2] = try f(iz)
			if hasColor {
				let rv = try f(ir!), gv = try f(ig!), bv = try f(ib!)
				let isFloat = props[ir!].type == .float32 || props[ir!].type == .float64
				let s: Float = isFloat ? 255.0 : 1.0
				pts.rgb[i * 3 + 0] = clampToUInt8(rv * s)
				pts.rgb[i * 3 + 1] = clampToUInt8(gv * s)
				pts.rgb[i * 3 + 2] = clampToUInt8(bv * s)
			}
		}
	}
	return pts
}

// MARK: - PLY point cloud

private enum PlyType {
	case float32, float64, uint8, uint16, int32, unknown
	
	init(_ t: String) {
		switch t {
			case "float", "float32": self = .float32
			case "double", "float64": self = .float64
			case "uchar", "uint8": self = .uint8
			case "ushort", "uint16": self = .uint16
			case "int", "int32": self = .int32
			default: self = .unknown
		}
	}
	
	var size: Int {
		switch self {
			case .float32, .int32: return 4
			case .float64: return 8
			case .uint8: return 1
			case .uint16: return 2
			case .unknown: return 0
		}
	}
}

private struct PlyProp {
	var name: String
	var type: PlyType
	var offset: Int
}

/// Splits a PLY file into header lines and the byte offset where the body starts.
public func plyHeader(_ data: Data) -> (lines: [String], bodyOffset: Int)? {
	var lines: [String] = []
	var start = 0
	let bytes = [UInt8](data.prefix(64 * 1024 * 16))  // headers are small
	var i = 0
	while i < bytes.count {
		if bytes[i] == 0x0A {
			var line = String(decoding: bytes[start..<i], as: UTF8.self)
			if line.hasSuffix("\r") { line.removeLast() }
			lines.append(line)
			start = i + 1
			if line == "end_header" { return (lines, start) }
		}
		i += 1
	}
	return nil
}
