import Foundation
import ColmapData
import NerfStudioData
import PolycamData
import simd


/// Convert a 16-element Float array (row-major) to simd_float4x4 (column-major).
/// This handles the conversion from the typical row-major format used in JSON files
/// to the column-major format used by SIMD types.
private func arrayToMatrix4x4(_ array: [Float]) -> simd_float4x4 {
	guard array.count == 16 else {
		// Return identity matrix if array is invalid
		return matrix_identity_float4x4
	}
	
	// Array is in row-major order: [m00, m01, m02, m03, m10, m11, m12, m13, ...]
	// simd_float4x4 expects column-major, so we need to transpose
	return simd_float4x4(
		simd_float4(array[0], array[4], array[8],  array[12]),  // column 0
		simd_float4(array[1], array[5], array[9],  array[13]),  // column 1
		simd_float4(array[2], array[6], array[10], array[14]),  // column 2
		simd_float4(array[3], array[7], array[11], array[15])   // column 3
	)
}


func fileExists(_ path: String) -> Bool { FileManager.default.fileExists(atPath: path) }

func joinPath(_ a: String, _ b: String) -> String {
    URL(fileURLWithPath: a).appendingPathComponent(b).path
}

private func loadJSON(_ path: String) throws -> Any {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONSerialization.jsonObject(with: data)
}

/// nlohmann-style `j.value(key, default)` for numbers.
private func num(_ j: [String: Any], _ key: String, _ def: Float) -> Float {
    (j[key] as? NSNumber)?.floatValue ?? def
}
private func num(_ j: [String: Any], _ key: String, _ def: Int) -> Int {
    (j[key] as? NSNumber)?.intValue ?? def
}

private func readMatrix4(_ any: Any?) -> [Float]? {
    guard let rows = any as? [[Any]], rows.count >= 4 else { return nil }
    var m = [Float](repeating: 0, count: 16)
    for r in 0..<4 {
        guard rows[r].count >= 4 else { return nil }
        for c in 0..<4 { m[r * 4 + c] = (rows[r][c] as? NSNumber)?.floatValue ?? 0 }
    }
    return m
}


// MARK: - Nerfstudio

/// Try common image extensions if the file doesn't exist.
private func resolveImagePath(_ path: String) -> String {
    if fileExists(path) { return path }
    for ext in [".png", ".jpg", ".jpeg", ".JPG"] where fileExists(path + ext) { return path + ext }
    return path
}





/// Adapter to convert NerfStudioData package types to MsplatCore InputData
extension NerfstudioDatasetSource : SplatDatasetSource
{
	public static func Detect(rootPath: URL) -> Bool {
		let fm = FileManager.default
		let transformsPath = rootPath.appendingPathComponent("transforms.json").path
		return fm.fileExists(atPath: transformsPath)
	}
	
	public func Load() async throws -> InputData {
		// Load the dataset using the NerfStudioData package
		let nerfDataset = try await loadDataset()
		
		// Convert to InputData
		return try convertToInputData(nerfDataset)
	}
	
	/// Convert NerfStudioDataset to MsplatCore's InputData format
	private func convertToInputData(_ nerfDataset: NerfStudioDataset) throws -> InputData {
		let inputData = InputData()
		
		// Convert cameras
		for nerfCamera in nerfDataset.cameras {
			var cameraMeta = CameraMeta(imageSource: .url(
				nerfDataset.baseDirectory.appendingPathComponent(nerfCamera.imagePath)
			))
			
			cameraMeta.intrinsics.width = nerfCamera.width
			cameraMeta.intrinsics.height = nerfCamera.height
			cameraMeta.intrinsics.fx = nerfCamera.fx
			cameraMeta.intrinsics.fy = nerfCamera.fy
			cameraMeta.intrinsics.cx = nerfCamera.cx
			cameraMeta.intrinsics.cy = nerfCamera.cy
			
			// Map distortion parameters (NerfStudio uses k1-k4, p1-p2; we use k1-k3, p1-p2)
			cameraMeta.distortion.k1 = nerfCamera.k1
			cameraMeta.distortion.k2 = nerfCamera.k2
			cameraMeta.distortion.k3 = nerfCamera.k3  // k4 is ignored - not supported by our distortion model
			cameraMeta.distortion.p1 = nerfCamera.p1
			cameraMeta.distortion.p2 = nerfCamera.p2
			
			// Convert transform matrix from [Float] array to simd_float4x4 (column-major)
			cameraMeta.camToWorld = arrayToMatrix4x4(nerfCamera.transformMatrix)
			
			inputData.cameras.append(cameraMeta)
		}
		
		// Convert point cloud if present
		if let pointCloud = nerfDataset.pointCloud {
			inputData.points.xyz = pointCloud.positions
			inputData.points.rgb = pointCloud.colors
			inputData.points.count = pointCloud.count
		}
		
		// Apply scale from the dataset
		inputData.scale = nerfDataset.appliedScale
		
		return inputData
	}
}





extension ColmapDatasetSource: SplatDatasetSource 
{
	public static func Detect(rootPath: URL) -> Bool {
		let fm = FileManager.default
		
		// Check for cameras.bin in root directory
		let camerasInRoot = rootPath.appendingPathComponent("cameras.bin").path
		if fm.fileExists(atPath: camerasInRoot) {
			return true
		}
		
		// Check for cameras.bin in sparse/0 directory (common COLMAP structure)
		let camerasInSparse = rootPath
			.appendingPathComponent("sparse")
			.appendingPathComponent("0")
			.appendingPathComponent("cameras.bin")
			.path
		return fm.fileExists(atPath: camerasInSparse)
	}
	
	public func Load() async throws -> InputData {
		// Load the dataset using the ColmapData package
		let colmapDataset = try await loadDataset()
		
		// Convert to InputData
		return try convertToInputData(colmapDataset)
	}
	
	/// Convert ColmapDataset to MsplatCore's InputData format
	private func convertToInputData(_ colmapDataset: ColmapDataset) throws -> InputData {
		let inputData = InputData()
		
		// Convert cameras
		for colmapCamera in colmapDataset.cameras 
		{
			var cameraMeta = CameraMeta(imageSource: .url(colmapCamera.imagePath))
			
			cameraMeta.intrinsics.width = colmapCamera.width
			cameraMeta.intrinsics.height = colmapCamera.height
			cameraMeta.intrinsics.fx = colmapCamera.fx
			cameraMeta.intrinsics.fy = colmapCamera.fy
			cameraMeta.intrinsics.cx = colmapCamera.cx
			cameraMeta.intrinsics.cy = colmapCamera.cy
			
			// Map distortion parameters
			cameraMeta.distortion.k1 = colmapCamera.k1
			cameraMeta.distortion.k2 = colmapCamera.k2
			cameraMeta.distortion.k3 = colmapCamera.k3
			cameraMeta.distortion.p1 = colmapCamera.p1
			cameraMeta.distortion.p2 = colmapCamera.p2
			
			// Convert transform matrix from [Float] array to simd_float4x4 (column-major)
			cameraMeta.camToWorld = simd_float4x4(colMajor: colmapCamera.transformMatrix)
			
			inputData.cameras.append(cameraMeta)
		}
		
		// Convert point cloud if present
		if let pointCloud = colmapDataset.pointCloud {
			inputData.points.xyz = pointCloud.positions
			inputData.points.rgb = pointCloud.colors
			inputData.points.count = pointCloud.count
		}
		
		// Apply scale from the dataset
		inputData.scale = colmapDataset.appliedScale
		
		return inputData
	}
}




extension PolycamDatasetSource: SplatDatasetSource 
{
	public static func Detect(rootPath: URL) -> Bool 
	{
		let fm = FileManager.default
		
		// Check for keyframes/corrected_cameras
		let correctedCameras = rootPath
			.appendingPathComponent("keyframes")
			.appendingPathComponent("corrected_cameras")
			.path
		if fm.fileExists(atPath: correctedCameras) {
			return true
		}
		
		// Check for cameras.json
		let camerasJson = rootPath.appendingPathComponent("cameras.json").path
		return fm.fileExists(atPath: camerasJson)
	}
	
	public func Load() async throws -> InputData 
	{
		// Load the dataset using the PolycamData package
		let polycamDataset = try await loadDataset()
		
		// Convert to InputData
		return try convertToInputData(polycamDataset)
	}
	
	/// Convert PolycamDataset to MsplatCore's InputData format
	private func convertToInputData(_ polycamDataset: PolycamDataset) throws -> InputData {
		let inputData = InputData()
		
		// Convert cameras
		for polycamCamera in polycamDataset.cameras {
			var cameraMeta = CameraMeta(imageSource: .url(
				polycamDataset.baseDirectory.appendingPathComponent(polycamCamera.imagePath)
			))
			
			cameraMeta.intrinsics.width = polycamCamera.width
			cameraMeta.intrinsics.height = polycamCamera.height
			cameraMeta.intrinsics.fx = polycamCamera.fx
			cameraMeta.intrinsics.fy = polycamCamera.fy
			cameraMeta.intrinsics.cx = polycamCamera.cx
			cameraMeta.intrinsics.cy = polycamCamera.cy
			
			// Map distortion parameters (Polycam uses k1, k2, p1, p2)
			cameraMeta.distortion.k1 = polycamCamera.k1
			cameraMeta.distortion.k2 = polycamCamera.k2
			cameraMeta.distortion.k3 = 0  // Not provided by Polycam
			cameraMeta.distortion.p1 = polycamCamera.p1
			cameraMeta.distortion.p2 = polycamCamera.p2
			
			// Convert transform matrix from [Float] array to simd_float4x4 (column-major)
			cameraMeta.camToWorld = arrayToMatrix4x4(polycamCamera.transformMatrix)
			
			inputData.cameras.append(cameraMeta)
		}
		
		// Convert point cloud if present
		if let pointCloud = polycamDataset.pointCloud {
			inputData.points.xyz = pointCloud.positions
			inputData.points.rgb = pointCloud.colors
			inputData.points.count = pointCloud.count
		}
		
		// Apply scale from the dataset
		inputData.scale = polycamDataset.appliedScale
		
		return inputData
	}
}

extension PolycamIntrinsics
{
	/// Convert to CameraMeta, applying defaults where needed
	func toCameraMeta(imageSource: ImageSource) -> CameraMeta {
		var meta = CameraMeta(imageSource: imageSource)
		let w = width ?? 0
		let h = height ?? 0
		meta.intrinsics.width = w
		meta.intrinsics.height = h
		meta.intrinsics.fx = Float(fx ?? 0)
		meta.intrinsics.fy = Float(fy ?? 0)
		meta.intrinsics.cx = Float(cx ?? Double(w) / 2.0)
		meta.intrinsics.cy = Float(cy ?? Double(h) / 2.0)
		return meta
	}
}
