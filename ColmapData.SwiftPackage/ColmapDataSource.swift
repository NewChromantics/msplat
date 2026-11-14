import Foundation




/// Represents optional point cloud data in a COLMAP dataset
public struct ColmapPointCloud {
	/// XYZ positions (count * 3 floats)
	public var positions: [Float]
	/// RGB colors (count * 3 bytes)
	public var colors: [UInt8]
	/// Number of points
	public var count: Int
	
	public init(positions: [Float], colors: [UInt8], count: Int) {
		self.positions = positions
		self.colors = colors
		self.count = count
	}
}

/// Complete COLMAP dataset loaded from cameras.bin, images.bin, and points3D.bin
public struct ColmapDataset {
	/// All cameras in the dataset
	public var cameras: [ColmapCameraWithTransform]
	/// Optional point cloud data
	public var pointCloud: ColmapPointCloud?
	/// Base directory path for resolving relative image paths
	public var baseDirectory: URL
	/// Applied coordinate system scale (if any)
	public var appliedScale: Float
	
	public init(
		cameras: [ColmapCameraWithTransform] = [],
		pointCloud: ColmapPointCloud? = nil,
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

/// Source for loading COLMAP datasets
public struct ColmapDatasetSource
{
	let url: URL
	let explicitImageSourcePath: URL?
	var imageSourcePath: URL	{	explicitImageSourcePath ?? url.appending(component: "images")	}
	
	public init(url: URL,imageSourcePath:URL?)
	{
		self.url = url
		self.explicitImageSourcePath = imageSourcePath
	}
	
	/// Load a COLMAP dataset from a directory containing cameras.bin, images.bin
	/// - Returns: A ColmapDataset with camera metadata and optional point cloud
	/// - Throws: If the required binary files cannot be found or parsed
	public func loadDataset() async throws -> ColmapDataset {
		// Find the sparse reconstruction directory
		let sparseURL = findSparseDirectory()
		
		// Read cameras.bin
		let camerasURL = sparseURL.appendingPathComponent("cameras.bin")
		let camerasData = try readCamerasBin(camerasURL)
		
		// Read images.bin
		let imagesURL = sparseURL.appendingPathComponent("images.bin")
		let imagesData = try readImagesBin(imagesURL)
		
		// Merge cameras and images to create ColmapCamera instances
		var cameras: [ColmapCameraWithTransform] = []
		for imageData in imagesData {
			guard let cameraData = camerasData.first(where: { $0.id == imageData.cameraID }) else {
				continue
			}
			
			// Convert COLMAP world-to-camera to camera-to-world (OpenGL)
			let transform = convertWorldToCameraToCameraToWorld(
				quat: imageData.quat,
				translation: imageData.translation
			)
			
			// Determine image path
			let imagePath = imageSourcePath.appending(component: imageData.name)
			
			let camera = ColmapCameraWithTransform(
				id: imageData.id,
				width: cameraData.width,
				height: cameraData.height,
				fx: cameraData.fx,
				fy: cameraData.fy,
				cx: cameraData.cx,
				cy: cameraData.cy,
				k1: cameraData.k1,
				k2: cameraData.k2,
				k3: cameraData.k3,
				p1: cameraData.p1,
				p2: cameraData.p2,
				transformMatrix: transform,
				imagePath: imagePath
			)
			
			cameras.append(camera)
		}
		
		// Try to load point cloud from points3D.bin (optional)
		var pointCloud: ColmapPointCloud? = nil
		let pointsURL = sparseURL.appendingPathComponent("points3D.bin")
		if FileManager.default.fileExists(atPath: pointsURL.path) {
			pointCloud = try readPoints3DBin(pointsURL)
		}
		
		return ColmapDataset(
			cameras: cameras,
			pointCloud: pointCloud,
			baseDirectory: url,
			appliedScale: 1.0
		)
	}
	
	private func findSparseDirectory() -> URL {
		// Check if cameras.bin is in the root
		if FileManager.default.fileExists(atPath: url.appendingPathComponent("cameras.bin").path) {
			return url
		}
		
		// Check sparse/0 directory (common structure)
		let sparse0 = url.appendingPathComponent("sparse").appendingPathComponent("0")
		if FileManager.default.fileExists(atPath: sparse0.appendingPathComponent("cameras.bin").path) {
			return sparse0
		}
		
		// Check sparse directory
		let sparse = url.appendingPathComponent("sparse")
		if FileManager.default.fileExists(atPath: sparse.appendingPathComponent("cameras.bin").path) {
			return sparse
		}
		
		return url
	}
	
	private struct CameraData {
		var id: Int
		var model: ColmapCameraModel
		var width: Int
		var height: Int
		var fx: Float
		var fy: Float
		var cx: Float
		var cy: Float
		var k1: Float
		var k2: Float
		var k3: Float
		var p1: Float
		var p2: Float
	}
	
	private struct ImageData {
		var id: Int
		var cameraID: Int
		var name: String
		var quat: [Double]  // Quaternion (w, x, y, z)
		var translation: [Double]  // (tx, ty, tz)
	}
	
	private func readCamerasBin(_ url: URL) throws -> [CameraData] {
		let data = try Data(contentsOf: url)
		var offset = 0
		
		// Read number of cameras
		guard data.count >= 8 else 
		{
			throw ColmapError("Not enough cameras \(data.count)/8")
		}
		let numCameras = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
		offset += 8
		
		var cameras: [CameraData] = []
		
		for _ in 0..<numCameras {
			guard offset + 24 <= data.count else { throw ColmapError("Bad data (camera meta)") }
			
			let cameraID = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
			offset += 4
			
			let modelID = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
			offset += 4
			
			let width = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
			offset += 8
			
			let height = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
			offset += 8
			
			guard let model = ColmapCameraModel(rawValue: modelID) else 
			{
				throw ColmapError("unsupportedCameraModel \(modelID)")
			}
			
			// Read parameters (variable length based on model)
			let numParams = getNumParams(for: model)
			guard offset + numParams * 8 <= data.count else { throw ColmapError("Bad data (params)") }
			
			var params: [Double] = []
			for _ in 0..<numParams {
				let param = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Double.self) }
				params.append(param)
				offset += 8
			}
			
			// Extract intrinsics based on model type
			let intrinsics = extractIntrinsics(model: model, params: params, width: Int(width), height: Int(height))
			
			cameras.append(CameraData(
				id: Int(cameraID),
				model: model,
				width: Int(width),
				height: Int(height),
				fx: intrinsics.fx,
				fy: intrinsics.fy,
				cx: intrinsics.cx,
				cy: intrinsics.cy,
				k1: intrinsics.k1,
				k2: intrinsics.k2,
				k3: intrinsics.k3,
				p1: intrinsics.p1,
				p2: intrinsics.p2
			))
		}
		
		return cameras
	}
	
	private func readImagesBin(_ url: URL) throws -> [ImageData] {
		let data = try Data(contentsOf: url)
		var reader = BinaryReader(data)
		
		// Read number of images
		let numImages = try reader.read(UInt64.self)
		
		var images: [ImageData] = []
		images.reserveCapacity(Int(numImages))
		
		for _ in 0..<numImages {
			let imageID = try reader.read(UInt32.self)
			
			// Read quaternion (w, x, y, z)
			let qw = try reader.readF64()
			let qx = try reader.readF64()
			let qy = try reader.readF64()
			let qz = try reader.readF64()
			
			// Read translation
			let tx = try reader.readF64()
			let ty = try reader.readF64()
			let tz = try reader.readF64()
			
			let cameraID = try reader.read(UInt32.self)
			
			// Read null-terminated image name
			let name = try reader.readCString()
			
			// Skip point2D data (not needed for camera reconstruction)
			let numPoints2D = try reader.read(UInt64.self)
			try reader.skip(Int(numPoints2D) * 24)  // Each point2D is 24 bytes (x, y, point3D_id)
			
			images.append(ImageData(
				id: Int(imageID),
				cameraID: Int(cameraID),
				name: name,
				quat: [qw, qx, qy, qz],
				translation: [tx, ty, tz]
			))
		}
		
		return images
	}
	
	private func readPoints3DBin(_ url: URL) throws -> ColmapPointCloud {
		let data = try Data(contentsOf: url)
		var reader = BinaryReader(data)
		
		// Read number of points
		let numPoints = try reader.read(UInt64.self)
		
		var positions: [Float] = []
		var colors: [UInt8] = []
		
		positions.reserveCapacity(Int(numPoints) * 3)
		colors.reserveCapacity(Int(numPoints) * 3)
		
		for _ in 0..<numPoints {
			// Skip point3D ID
			_ = try reader.read(UInt64.self)
			
			// Read XYZ
			let x = try reader.readF64()
			let y = try reader.readF64()
			let z = try reader.readF64()
			
			// Read RGB
			let r = try reader.read(UInt8.self)
			let g = try reader.read(UInt8.self)
			let b = try reader.read(UInt8.self)
			
			// Skip error
			_ = try reader.readF64()
			
			// Skip track data
			let trackLength = try reader.read(UInt64.self)
			try reader.skip(Int(trackLength) * 8)  // Each track element is 8 bytes
			
			positions.append(contentsOf: [Float(x), Float(y), Float(z)])
			colors.append(contentsOf: [r, g, b])
		}
		
		return ColmapPointCloud(
			positions: positions,
			colors: colors,
			count: positions.count / 3
		)
	}
	
	private func getNumParams(for model: ColmapCameraModel) -> Int {
		switch model {
		case .simplePinhole: return 3  // f, cx, cy
		case .pinhole: return 4  // fx, fy, cx, cy
		case .simpleRadial: return 4  // f, cx, cy, k
		case .radial: return 5  // f, cx, cy, k1, k2
		case .opencv: return 8  // fx, fy, cx, cy, k1, k2, p1, p2
		case .fullOpencv: return 12  // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
		case .simpleRadialFisheye: return 4
		case .radialFisheye: return 5
		case .thinPrismFisheye: return 12
		}
	}
	
	private func extractIntrinsics(model: ColmapCameraModel, params: [Double], width: Int, height: Int) 
		-> (fx: Float, fy: Float, cx: Float, cy: Float, k1: Float, k2: Float, k3: Float, p1: Float, p2: Float) {
		
		switch model {
		case .simplePinhole:
			let f = Float(params[0])
			return (f, f, Float(params[1]), Float(params[2]), 0, 0, 0, 0, 0)
			
		case .pinhole:
			return (Float(params[0]), Float(params[1]), Float(params[2]), Float(params[3]), 0, 0, 0, 0, 0)
			
		case .simpleRadial:
			let f = Float(params[0])
			return (f, f, Float(params[1]), Float(params[2]), Float(params[3]), 0, 0, 0, 0)
			
		case .radial:
			let f = Float(params[0])
			return (f, f, Float(params[1]), Float(params[2]), Float(params[3]), Float(params[4]), 0, 0, 0)
			
		case .opencv:
			return (Float(params[0]), Float(params[1]), Float(params[2]), Float(params[3]),
					Float(params[4]), Float(params[5]), 0, Float(params[6]), Float(params[7]))
			
		case .fullOpencv:
			return (Float(params[0]), Float(params[1]), Float(params[2]), Float(params[3]),
					Float(params[4]), Float(params[5]), Float(params[8]), Float(params[6]), Float(params[7]))
			
		default:
			// Fallback for unsupported models
			return (Float(width) * 0.5, Float(height) * 0.5, Float(width) * 0.5, Float(height) * 0.5, 0, 0, 0, 0, 0)
		}
	}
	
	private func convertWorldToCameraToCameraToWorld(quat: [Double], translation: [Double]) -> [Float] {
		// COLMAP stores world-to-camera transform as quaternion + translation
		// We need to convert to camera-to-world (4x4 row-major, OpenGL convention: Y-up, Z-back)
		
		// Normalize quaternion first (important!)
		var qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3]
		let n = (qw * qw + qx * qx + qy * qy + qz * qz).squareRoot()
		qw /= n; qx /= n; qy /= n; qz /= n
		
		// Build rotation matrix from quaternion (world-to-camera)
		let r11 = 1.0 - 2.0 * (qy * qy + qz * qz)
		let r12 = 2.0 * (qx * qy - qw * qz)
		let r13 = 2.0 * (qx * qz + qw * qy)
		
		let r21 = 2.0 * (qx * qy + qw * qz)
		let r22 = 1.0 - 2.0 * (qx * qx + qz * qz)
		let r23 = 2.0 * (qy * qz - qw * qx)
		
		let r31 = 2.0 * (qx * qz - qw * qy)
		let r32 = 2.0 * (qy * qz + qw * qx)
		let r33 = 1.0 - 2.0 * (qx * qx + qy * qy)
		
		// Invert rotation (transpose for orthogonal matrix)
		// Camera-to-world position: -R^T * t
		let cx = -(r11 * translation[0] + r21 * translation[1] + r31 * translation[2])
		let cy = -(r12 * translation[0] + r22 * translation[1] + r32 * translation[2])
		let cz = -(r13 * translation[0] + r23 * translation[1] + r33 * translation[2])
		
		// Build camera-to-world matrix (row-major) with transposed rotation and OpenGL Y/Z flip
		// COLMAP uses Y-down, Z-forward; OpenGL uses Y-up, Z-back
		// So we transpose the rotation (r11->r11, r12->r21, r13->r31, etc.) and negate Y and Z columns
		return [
			Float(r11), Float(-r21), Float(-r31), Float(cx),
			Float(r12), Float(-r22), Float(-r32), Float(cy),
			Float(r13), Float(-r23), Float(-r33), Float(cz),
			0, 0, 0, 1
		]
	}
}
