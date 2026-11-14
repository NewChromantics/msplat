import Foundation

// MARK: - JSON Decoding Structures

/// Represents a camera entry in cameras.json
private struct PolycamCameraJSON: Codable {
	let id: Int
	let img: String
	let width: Int
	let height: Int
	let fx: Double
	let fy: Double
	let cx: Double
	let cy: Double
	let t: [Double]  // translation (3 values)
	let r: [[Double]]  // rotation matrix (3x3)
	let k1: Double?
	let k2: Double?
	let p1: Double?
	let p2: Double?
}

/// Represents the cameras.json file format
private struct PolycamCamerasJSON: Codable {
	let cameras: [PolycamCameraJSON]
}

// MARK: - Public API

/// Represents a single camera in a Polycam dataset
public struct PolycamCamera {
	public var id: Int
	public var width: Int
	public var height: Int
	public var fx: Float
	public var fy: Float
	public var cx: Float
	public var cy: Float
	public var k1: Float
	public var k2: Float
	public var p1: Float
	public var p2: Float
	/// 4x4 row-major camera-to-world transform matrix (OpenGL convention: Y-up, Z-back)
	public var transformMatrix: [Float]
	/// Relative path to the image file
	public var imagePath: String
	
	public init(
		id: Int,
		width: Int,
		height: Int,
		fx: Float,
		fy: Float,
		cx: Float,
		cy: Float,
		k1: Float = 0,
		k2: Float = 0,
		p1: Float = 0,
		p2: Float = 0,
		transformMatrix: [Float],
		imagePath: String
	) {
		self.id = id
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.k1 = k1
		self.k2 = k2
		self.p1 = p1
		self.p2 = p2
		self.transformMatrix = transformMatrix
		self.imagePath = imagePath
	}
}


/// Complete Polycam dataset loaded from cameras.json or keyframes/corrected_cameras
public struct PolycamDataset {
	/// All cameras in the dataset
	public var cameras: [PolycamCamera]
	/// Optional point cloud data
	public var pointCloud: PolycamPointCloud?
	/// Base directory path for resolving relative image paths
	public var baseDirectory: URL
	/// Applied coordinate system scale (if any)
	public var appliedScale: Float
	
	public init(
		cameras: [PolycamCamera] = [],
		pointCloud: PolycamPointCloud? = nil,
		baseDirectory: URL,
		appliedScale: Float = 1.0
	) {
		self.cameras = cameras
		self.pointCloud = pointCloud
		self.baseDirectory = baseDirectory
		self.appliedScale = appliedScale
	}
}

// MARK: - Dataset Source

/// Source for loading Polycam datasets
public struct PolycamDatasetSource {
	private let url: URL
	
	public init(url: URL) {
		self.url = url
	}
	
	/// Load a Polycam dataset from a directory containing cameras.json or keyframes/corrected_cameras
	/// - Returns: A PolycamDataset with camera metadata and optional point cloud
	/// - Throws: If the dataset files cannot be found or parsed
	public func loadDataset() async throws -> PolycamDataset {
		// Try cameras.json first
		let camerasJsonURL = url.appendingPathComponent("cameras.json")
		if FileManager.default.fileExists(atPath: camerasJsonURL.path) {
			return try await loadFromCamerasJSON(camerasJsonURL)
		}
		
		// Try keyframes/corrected_cameras format
		let keyframesURL = url.appendingPathComponent("keyframes")
		let correctedCamerasURL = keyframesURL.appendingPathComponent("corrected_cameras")
		if FileManager.default.fileExists(atPath: correctedCamerasURL.path) {
			return try await loadFromCorrectedCameras(keyframesURL)
		}
		
		throw PolycamError.noValidDatasetFound(url.path)
	}
	
	private func loadFromCamerasJSON(_ camerasURL: URL) async throws -> PolycamDataset {
		let data = try Data(contentsOf: camerasURL)
		let decoder = JSONDecoder()
		let camerasJSON = try decoder.decode(PolycamCamerasJSON.self, from: data)
		
		var cameras: [PolycamCamera] = []
		
		for camJSON in camerasJSON.cameras {
			// Build 4x4 camera-to-world matrix from rotation and translation
			// Polycam format: R is 3x3 rotation, t is translation
			let r = camJSON.r
			let t = camJSON.t
			
			// Convert to 4x4 row-major matrix (OpenGL convention)
			let transform: [Float] = [
				Float(r[0][0]), Float(r[0][1]), Float(r[0][2]), Float(t[0]),
				Float(r[1][0]), Float(r[1][1]), Float(r[1][2]), Float(t[1]),
				Float(r[2][0]), Float(r[2][1]), Float(r[2][2]), Float(t[2]),
				0, 0, 0, 1
			]
			
			let camera = PolycamCamera(
				id: camJSON.id,
				width: camJSON.width,
				height: camJSON.height,
				fx: Float(camJSON.fx),
				fy: Float(camJSON.fy),
				cx: Float(camJSON.cx),
				cy: Float(camJSON.cy),
				k1: Float(camJSON.k1 ?? 0),
				k2: Float(camJSON.k2 ?? 0),
				p1: Float(camJSON.p1 ?? 0),
				p2: Float(camJSON.p2 ?? 0),
				transformMatrix: transform,
				imagePath: camJSON.img
			)
			
			cameras.append(camera)
		}
		
		return PolycamDataset(
			cameras: cameras,
			pointCloud: nil,
			baseDirectory: url,
			appliedScale: 1.0
		)
	}
	
	private func loadFromCorrectedCameras(_ keyframesURL: URL) async throws -> PolycamDataset {
		// Parse corrected_cameras file (text format)
		let correctedCamerasURL = keyframesURL.appendingPathComponent("corrected_cameras")
		let content = try String(contentsOf: correctedCamerasURL, encoding: .utf8)
		let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
		
		var cameras: [PolycamCamera] = []
		var currentID = 0
		
		for line in lines {
			// Parse camera parameters from corrected_cameras format
			// Format varies, but typically includes intrinsics and extrinsics
			// This is a simplified parser - adjust based on actual format
			let components = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
			
			guard components.count >= 17 else { continue }
			
			// Parse intrinsics (fx, fy, cx, cy)
			guard let fx = Float(components[0]),
				  let fy = Float(components[1]),
				  let cx = Float(components[2]),
				  let cy = Float(components[3]) else { continue }
			
			// Parse extrinsics (4x4 matrix or 3x4 projection matrix)
			var transform = [Float](repeating: 0, count: 16)
			for i in 0..<12 {
				guard let value = Float(components[4 + i]) else { continue }
				let row = i / 4
				let col = i % 4
				transform[row * 4 + col] = value
			}
			transform[15] = 1.0  // Homogeneous coordinate
			
			// Determine image path
			let imagePath = "keyframes/images/frame_\(String(format: "%05d", currentID)).jpg"
			
			let camera = PolycamCamera(
				id: currentID,
				width: 0,  // Will be determined from actual image
				height: 0,  // Will be determined from actual image
				fx: fx,
				fy: fy,
				cx: cx,
				cy: cy,
				transformMatrix: transform,
				imagePath: imagePath
			)
			
			cameras.append(camera)
			currentID += 1
		}
		
		return PolycamDataset(
			cameras: cameras,
			pointCloud: nil,
			baseDirectory: url,
			appliedScale: 1.0
		)
	}
}

// MARK: - Errors

enum PolycamError: Error, CustomStringConvertible {
	case noValidDatasetFound(String)
	
	var description: String {
		switch self {
		case .noValidDatasetFound(let path):
			return "No valid Polycam dataset found in: \(path)\n" +
				   "Expected cameras.json or keyframes/corrected_cameras"
		}
	}
}


// MARK: - Intrinsics Decoding

/// Represents camera intrinsics in Polycam format
public struct PolycamIntrinsics: Codable 
{
	public let width: Int?
	public let height: Int?
	public let fx: Double?
	public let fy: Double?
	public let cx: Double?
	public let cy: Double?
}


/// Represents individual camera JSON in keyframes/corrected_cameras (Layout 1)
private struct PolycamCorrectedCameraJSON: Codable {
	let width: Int?
	let height: Int?
	let fx: Double?
	let fy: Double?
	let cx: Double?
	let cy: Double?
	
	// Rotation matrix elements (3x3)
	let R_00: Double?
	let R_01: Double?
	let R_02: Double?
	let R_10: Double?
	let R_11: Double?
	let R_12: Double?
	let R_20: Double?
	let R_21: Double?
	let R_22: Double?
	
	// Translation vector
	let t_20: Double?  // t[0]
	let t_21: Double?  // t[1]
	let t_22: Double?  // t[2]
}

/// Represents frame entry in cameras.json (Layout 2)
private struct PolycamFrameJSON: Codable {
	let width: Int?
	let height: Int?
	let fx: Double?
	let fy: Double?
	let cx: Double?
	let cy: Double?
	let file_path: String?
	let transform_matrix: [[Double]]?
}

/// Represents cameras.json with frames array (Layout 2)
private struct PolycamCamerasWithFramesJSON: Codable {
	let frames: [PolycamFrameJSON]
}

/// Helper to convert 4x4 matrix array to flat array
private func readMatrix4(_ matrix: [[Double]]?) -> [Float]? {
	guard let m = matrix, m.count == 4 else { return nil }
	var result = [Float](repeating: 0, count: 16)
	for row in 0..<4 {
		guard m[row].count == 4 else { return nil }
		for col in 0..<4 {
			result[row * 4 + col] = Float(m[row][col])
		}
	}
	return result
}

/*
private func findImage(_ dir: String, _ stem: String) -> String {
	for ext in [".png", ".jpg", ".jpeg", ".JPG"] {
		let p = joinPath(dir, stem + ext)
		if fileExists(p) { return p }
	}
	return joinPath(dir, stem + ".png")
}

func loadPolycam(_ projectRoot: String) throws -> InputData {
	let data = InputData()
	let keyframesDir = joinPath(joinPath(projectRoot, "keyframes"), "corrected_cameras")
	let imagesDir = joinPath(joinPath(projectRoot, "keyframes"), "corrected_images")
	
	let decoder = JSONDecoder()
	
	if fileExists(keyframesDir) {
		// Layout 1: individual camera JSONs
		let files = try FileManager.default.contentsOfDirectory(atPath: keyframesDir)
			.filter { ($0 as NSString).pathExtension == "json" }
			.map { joinPath(keyframesDir, $0) }
		
		for jp in files {
			let jsonData = try Data(contentsOf: URL(fileURLWithPath: jp))
			let camJSON = try decoder.decode(PolycamCorrectedCameraJSON.self, from: jsonData)
			
			// Build rotation matrix
			let R: [Float] = [
				Float(camJSON.R_00 ?? 0), Float(camJSON.R_01 ?? 0), Float(camJSON.R_02 ?? 0),
				Float(camJSON.R_10 ?? 0), Float(camJSON.R_11 ?? 0), Float(camJSON.R_12 ?? 0),
				Float(camJSON.R_20 ?? 0), Float(camJSON.R_21 ?? 0), Float(camJSON.R_22 ?? 0)
			]
			
			// Build translation vector
			let T: [Float] = [
				Float(camJSON.t_20 ?? 0),
				Float(camJSON.t_21 ?? 0),
				Float(camJSON.t_22 ?? 0)
			]
			
			// Convert to CameraMeta (initially with placeholder image source)
			let stem = ((jp as NSString).lastPathComponent as NSString).deletingPathExtension
			let filePath = findImage(imagesDir, stem)
			
			let intrinsics = PolycamIntrinsics(
				width: camJSON.width,
				height: camJSON.height,
				fx: camJSON.fx,
				fy: camJSON.fy,
				cx: camJSON.cx,
				cy: camJSON.cy
			)
			var meta = intrinsics.toCameraMeta(imageSource: .url(URL(fileURLWithPath: filePath)))
			
			// Set camera-to-world with OpenGL Y/Z flip
			meta.camToWorld = [
				R[0],  R[1],  R[2],  T[0],
				-R[3], -R[4], -R[5], -T[1],
				-R[6], -R[7], -R[8], -T[2],
				0,     0,     0,     1,
			]
			
			data.cameras.append(meta)
		}
	} else {
		// Layout 2: single cameras.json
		let camerasPath = joinPath(projectRoot, "cameras.json")
		let jsonData = try Data(contentsOf: URL(fileURLWithPath: camerasPath))
		
		// Try to decode as object with frames array first
		var frames: [PolycamFrameJSON] = []
		if let camerasWithFrames = try? decoder.decode(PolycamCamerasWithFramesJSON.self, from: jsonData) {
			frames = camerasWithFrames.frames
		} else if let framesArray = try? decoder.decode([PolycamFrameJSON].self, from: jsonData) {
			// Try as direct array
			frames = framesArray
		}
		
		for frameJSON in frames {
			let intrinsics = PolycamIntrinsics(
				width: frameJSON.width,
				height: frameJSON.height,
				fx: frameJSON.fx,
				fy: frameJSON.fy,
				cx: frameJSON.cx,
				cy: frameJSON.cy
			)
			
			var filePath = frameJSON.file_path ?? ""
			if !filePath.isEmpty && !filePath.hasPrefix("/") {
				filePath = joinPath(projectRoot, filePath)
			}
			
			var meta = intrinsics.toCameraMeta(imageSource: .url(URL(fileURLWithPath: filePath)))
			
			// Set transform matrix if available
			if let transform = readMatrix4(frameJSON.transform_matrix) {
				meta.camToWorld = transform
			}
			
			data.cameras.append(meta)
		}
	}
	
	for rel in ["keyframes/point_cloud.ply", "point_cloud.ply", "sparse.ply"] {
		let path = joinPath(projectRoot, rel)
		if fileExists(path) { data.points = try readPly(path); break }
	}
	
	autoScaleAndCenter(data)
	return data
}
*/
