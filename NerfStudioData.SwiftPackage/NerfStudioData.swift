import Foundation

// MARK: - JSON Decoding Structures

/// Represents the structure of transforms.json file
private struct TransformsJSON: Codable {
	let w: Int?
	let h: Int?
	let fl: Double?
	let flX: Double?
	let flY: Double?
	let cx: Double?
	let cy: Double?
	let k1: Double?
	let k2: Double?
	let k3: Double?
	let k4: Double?
	let p1: Double?
	let p2: Double?
	let appliedScale: Double?
	let plyFilePath: String?
	let frames: [FrameJSON]
	
	enum CodingKeys: String, CodingKey {
		case w, h, fl
		case flX = "fl_x"
		case flY = "fl_y"
		case cx, cy
		case k1, k2, k3, k4
		case p1, p2
		case appliedScale = "applied_scale"
		case plyFilePath = "ply_file_path"
		case frames
	}
}

/// Represents a single frame entry in transforms.json
private struct FrameJSON: Codable {
	let w: Int?
	let h: Int?
	let flX: Double?
	let flY: Double?
	let cx: Double?
	let cy: Double?
	let k1: Double?
	let k2: Double?
	let k3: Double?
	let k4: Double?
	let p1: Double?
	let p2: Double?
	let transformMatrix: [[Double]]
	let filePath: String?
	
	enum CodingKeys: String, CodingKey {
		case w, h
		case flX = "fl_x"
		case flY = "fl_y"
		case cx, cy
		case k1, k2, k3, k4
		case p1, p2
		case transformMatrix = "transform_matrix"
		case filePath = "file_path"
	}
}

// MARK: - Dataset Structure

/// Complete Nerfstudio dataset loaded from transforms.json
public struct NerfStudioDataset 
{
	/// All cameras in the dataset
	public var cameras: [NerfStudioCamera]
	/// Optional point cloud data
	public var pointCloud: NerfStudioPointCloud?
	/// Base directory path for resolving relative image paths
	public var baseDirectory: URL
	/// Applied coordinate system scale (if any)
	public var appliedScale: Float
	
	public init(
		cameras: [NerfStudioCamera] = [],
		pointCloud: NerfStudioPointCloud? = nil,
		baseDirectory: URL,
		appliedScale: Float = 1.0
	) {
		self.cameras = cameras
		self.pointCloud = pointCloud
		self.baseDirectory = baseDirectory
		self.appliedScale = appliedScale
	}
}



public struct NerfstudioDatasetSource 
{
	private let url: URL
	
	public init(url: URL)
	{
		self.url = url
	}
	
	/// Load a Nerfstudio dataset from a directory containing transforms.json
	/// - Returns: A NerfStudioDataset with camera metadata and optional point cloud
	/// - Throws: If the transforms.json file cannot be found or parsed
	public func loadDataset() async throws -> NerfStudioDataset {
		let transformsURL = url.appendingPathComponent("transforms.json")
		
		let data = try Data(contentsOf: transformsURL)
		let decoder = JSONDecoder()
		let transforms = try decoder.decode(TransformsJSON.self, from: data)
		
		return try parseTransformsJSON(transforms, baseURL: url)
	}
	
	private func parseTransformsJSON(_ transforms: TransformsJSON, baseURL: URL) throws -> NerfStudioDataset {
		// Extract global camera parameters (can be overridden per-frame)
		let globalW = transforms.w ?? 0
		let globalH = transforms.h ?? 0
		let globalFl = transforms.flX ?? transforms.fl ?? 0
		let globalFlX = Float(transforms.flX ?? globalFl)
		let globalFlY = Float(transforms.flY ?? globalFl)
		let globalCx = Float(transforms.cx ?? Double(globalW) * 0.5)
		let globalCy = Float(transforms.cy ?? Double(globalH) * 0.5)
		
		let globalK1 = Float(transforms.k1 ?? 0)
		let globalK2 = Float(transforms.k2 ?? 0)
		let globalK3 = Float(transforms.k3 ?? 0)
		let globalK4 = Float(transforms.k4 ?? 0)
		let globalP1 = Float(transforms.p1 ?? 0)
		let globalP2 = Float(transforms.p2 ?? 0)
		
		let appliedScale = Float(transforms.appliedScale ?? 1.0)
		
		// Parse frames
		var cameras: [NerfStudioCamera] = []
		
		for frame in transforms.frames {
			// Per-frame overrides
			let w = frame.w ?? globalW
			let h = frame.h ?? globalH
			let flX = Float(frame.flX ?? Double(globalFlX))
			let flY = Float(frame.flY ?? Double(globalFlY))
			let cx = Float(frame.cx ?? Double(globalCx))
			let cy = Float(frame.cy ?? Double(globalCy))
			
			let k1 = Float(frame.k1 ?? Double(globalK1))
			let k2 = Float(frame.k2 ?? Double(globalK2))
			let k3 = Float(frame.k3 ?? Double(globalK3))
			let k4 = Float(frame.k4 ?? Double(globalK4))
			let p1 = Float(frame.p1 ?? Double(globalP1))
			let p2 = Float(frame.p2 ?? Double(globalP2))
			
			// Parse transform matrix (4x4)
			let transformArray = frame.transformMatrix
			
			var transform = [Float](repeating: 0, count: 16)
			for row in 0..<min(4, transformArray.count) {
				for col in 0..<min(4, transformArray[row].count) {
					transform[row * 4 + col] = Float(transformArray[row][col])
				}
			}
			
			// Image path
			let imagePath = frame.filePath ?? ""
			
			let camera = NerfStudioCamera(
				width: w,
				height: h,
				fx: flX,
				fy: flY,
				cx: cx,
				cy: cy,
				k1: k1,
				k2: k2,
				k3: k3,
				k4: k4,
				p1: p1,
				p2: p2,
				transformMatrix: transform,
				imagePath: imagePath
			)
			
			cameras.append(camera)
		}
		
		// Parse point cloud data if present (ply_file_path)
		var pointCloud: NerfStudioPointCloud? = nil
		if let plyPath = transforms.plyFilePath {
			let plyURL = baseURL.appendingPathComponent(plyPath)
			if FileManager.default.fileExists(atPath: plyURL.path) {
				// TODO: Implement PLY loading
				throw NerfStudioError("Ply loading not implemented")
				//readPly()
			}
		}
		
		return NerfStudioDataset(
			cameras: cameras,
			pointCloud: pointCloud,
			baseDirectory: baseURL,
			appliedScale: appliedScale
		)
	}
}


