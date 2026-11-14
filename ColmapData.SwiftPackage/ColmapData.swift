import Foundation



public struct BinaryReaderError: LocalizedError 
{
	public var errorDescription: String?
	
	public init(_ description: String) 
	{
		self.errorDescription = description
	}
}


public struct Points {
	public var xyz: [Float] = []   // N*3
	public var rgb: [UInt8] = []   // N*3
	public var count = 0
	
	init() {}
	
	init(xyz: [Float], rgb: [UInt8], count: Int) {
		self.xyz = xyz
		self.rgb = rgb
		self.count = count
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


/// COLMAP points3D.bin reader.
func readColmapPoints(_ path: String) throws -> Points 
{
	guard let d = try? Data(contentsOf: URL(fileURLWithPath: path)) else 
	{
		throw ColmapError("Cannot open points3D.bin: \(path)")
	}
	var f = BinaryReader(d)
	let n = Int(try f.read(UInt64.self))
	var pts = Points()
	pts.count = n
	pts.xyz = [Float](repeating: 0, count: n * 3)
	pts.rgb = [UInt8](repeating: 0, count: n * 3)
	for i in 0..<n {
		_ = try f.read(UInt64.self) // point id
		pts.xyz[i * 3 + 0] = Float(try f.readF64())
		pts.xyz[i * 3 + 1] = Float(try f.readF64())
		pts.xyz[i * 3 + 2] = Float(try f.readF64())
		pts.rgb[i * 3 + 0] = try f.read(UInt8.self)
		pts.rgb[i * 3 + 1] = try f.read(UInt8.self)
		pts.rgb[i * 3 + 2] = try f.read(UInt8.self)
		_ = try f.readF64() // reprojection error
		let trackLen = Int(try f.read(UInt64.self))
		try f.skip(trackLen * 8) // (image_id u32, point2d_idx u32) per entry
	}
	return pts
}


private func readCamerasBin(_ path: String) throws -> [UInt32: ColmapCamera] {
	var f = BinaryReader(try Data(contentsOf: URL(fileURLWithPath: path)))
	let n = try f.read(UInt64.self)
	var cams: [UInt32: ColmapCamera] = [:]
	for _ in 0..<n {
		var c = ColmapCamera()
		c.id = try f.read(UInt32.self)
		c.model = Int(try f.read(UInt32.self))
		c.width = Int(try f.read(UInt64.self))
		c.height = Int(try f.read(UInt64.self))
		func rd() throws -> Float { Float(try f.readF64()) }
		switch c.model {
			case 0: // SIMPLE_PINHOLE
				c.fx = try rd(); c.fy = c.fx; c.cx = try rd(); c.cy = try rd()
			case 1: // PINHOLE
				c.fx = try rd(); c.fy = try rd(); c.cx = try rd(); c.cy = try rd()
			case 2: // SIMPLE_RADIAL
				c.fx = try rd(); c.fy = c.fx; c.cx = try rd(); c.cy = try rd(); c.k1 = try rd()
			case 3: // RADIAL
				c.fx = try rd(); c.fy = c.fx; c.cx = try rd(); c.cy = try rd(); c.k1 = try rd(); c.k2 = try rd()
			case 4: // OPENCV
				c.fx = try rd(); c.fy = try rd(); c.cx = try rd(); c.cy = try rd()
				c.k1 = try rd(); c.k2 = try rd(); c.p1 = try rd(); c.p2 = try rd()
			default:
				throw ColmapError("Unsupported COLMAP camera model: \(c.model)")
		}
		cams[c.id] = c
	}
	return cams
}

private func readImagesBin(_ path: String) throws -> [ColmapImage] {
	var f = BinaryReader(try Data(contentsOf: URL(fileURLWithPath: path)))
	let n = try f.read(UInt64.self)
	var images: [ColmapImage] = []
	images.reserveCapacity(Int(n))
	for _ in 0..<n {
		var img = ColmapImage()
		_ = try f.read(UInt32.self) // image id
		img.quat = [try f.readF64(), try f.readF64(), try f.readF64(), try f.readF64()]
		img.t = [try f.readF64(), try f.readF64(), try f.readF64()]
		img.camId = try f.read(UInt32.self)
		img.filename = try f.readCString()
		let numPts2D = try f.read(UInt64.self)
		try f.skip(Int(numPts2D) * 24)
		images.append(img)
	}
	return images
}

/// Quaternion [w,x,y,z] → row-major 3x3 rotation.
private func quatToRotMat(_ q: [Double]) -> [Float] {
	var w = q[0], x = q[1], y = q[2], z = q[3]
	let n = (w * w + x * x + y * y + z * z).squareRoot()
	w /= n; x /= n; y /= n; z /= n
	return [
		Float(1 - 2 * (y * y + z * z)), Float(2 * (x * y - w * z)), Float(2 * (x * z + w * y)),
		Float(2 * (x * y + w * z)), Float(1 - 2 * (x * x + z * z)), Float(2 * (y * z - w * x)),
		Float(2 * (x * z - w * y)), Float(2 * (y * z + w * x)), Float(1 - 2 * (x * x + y * y)),
	]
}

/// w2c rotation + translation → 4x4 c2w row-major with OpenGL Y/Z flip.
private func w2cToCamToWorld(_ quat: [Double], _ t: [Double]) -> [Float] {
	let R = quatToRotMat(quat)
	let Ri: [Float] = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]
	let t0 = Float(t[0]), t1 = Float(t[1]), t2 = Float(t[2])
	let Ti: [Float] = [
		-(Ri[0] * t0 + Ri[1] * t1 + Ri[2] * t2),
		 -(Ri[3] * t0 + Ri[4] * t1 + Ri[5] * t2),
		 -(Ri[6] * t0 + Ri[7] * t1 + Ri[8] * t2),
	]
	return [
		Ri[0], -Ri[1], -Ri[2], Ti[0],
		Ri[3], -Ri[4], -Ri[5], Ti[1],
		Ri[6], -Ri[7], -Ri[8], Ti[2],
		0, 0, 0, 1,
	]
}


public struct ColmapData
{
	public var cameras: [UInt32: (camera: ColmapCamera, image: ColmapImage, imageFilename: URL, camToWorld: [Float])]
	public var points: Points?
	
	public init(projectRoot: URL, imageSourcePath: URL?) throws 
	{
		let fileManager = FileManager.default
		
		// Determine sparse directory
		let camerasInRoot = projectRoot.appendingPathComponent("cameras.bin")
		let sparseDir: URL
		if fileManager.fileExists(atPath: camerasInRoot.path) {
			sparseDir = projectRoot
		} else {
			sparseDir = projectRoot
				.appendingPathComponent("sparse")
				.appendingPathComponent("0")
		}
		
		// Determine image directory
		let imageDir: URL
		if let imagePath = imageSourcePath {
			imageDir = imagePath
		} else {
			let imagesDir = projectRoot.appendingPathComponent("images")
			if fileManager.fileExists(atPath: imagesDir.path) {
				imageDir = imagesDir
			} else {
				imageDir = projectRoot
			}
		}
		
		// Read cameras and images
		let camerasBinPath = sparseDir.appendingPathComponent("cameras.bin").path
		let imagesBinPath = sparseDir.appendingPathComponent("images.bin").path
		
		let camerasDict = try readCamerasBin(camerasBinPath)
		let images = try readImagesBin(imagesBinPath)
		
		// Build camera dictionary
		var result: [UInt32: (camera: ColmapCamera, image: ColmapImage, imageFilename: URL, camToWorld: [Float])] = [:]
		
		for image in images {
			guard let camera = camerasDict[image.camId] else {
				throw ColmapError("Missing camera meta with id \(image.camId)")
			}
			
			let imageFilename = imageDir.appendingPathComponent(image.filename)
			let camToWorld = w2cToCamToWorld(image.quat, image.t)
			result[image.camId] = (camera, image, imageFilename, camToWorld)
		}
		
		self.cameras = result
		
		// Read points if available
		let ptsPath = sparseDir.appendingPathComponent("points3D.bin")
		let plyPath = sparseDir.appendingPathComponent("points3D.ply")
		
		if fileManager.fileExists(atPath: ptsPath.path) {
			self.points = try readColmapPoints(ptsPath.path)
		} else if fileManager.fileExists(atPath: plyPath.path) {
			// Note: readPly function is not defined in this file
			// You may need to implement it or remove this branch
			self.points = nil // Placeholder until readPly is implemented
		} else {
			self.points = nil
		}
	}
}
