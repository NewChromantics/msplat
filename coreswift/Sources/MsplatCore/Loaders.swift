import Foundation
import ColmapData
import NerfStudioData
import PolycamData



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
			
			cameraMeta.width = nerfCamera.width
			cameraMeta.height = nerfCamera.height
			cameraMeta.fx = nerfCamera.fx
			cameraMeta.fy = nerfCamera.fy
			cameraMeta.cx = nerfCamera.cx
			cameraMeta.cy = nerfCamera.cy
			
			// Map distortion parameters (NerfStudio uses k1-k4, p1-p2; we use k1-k3, p1-p2)
			cameraMeta.k1 = nerfCamera.k1
			cameraMeta.k2 = nerfCamera.k2
			cameraMeta.k3 = nerfCamera.k3  // k4 is ignored - not supported by our distortion model
			cameraMeta.p1 = nerfCamera.p1
			cameraMeta.p2 = nerfCamera.p2
			
			// Copy transform matrix (already in OpenGL convention)
			cameraMeta.camToWorld = nerfCamera.transformMatrix
			
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
			
			cameraMeta.width = colmapCamera.width
			cameraMeta.height = colmapCamera.height
			cameraMeta.fx = colmapCamera.fx
			cameraMeta.fy = colmapCamera.fy
			cameraMeta.cx = colmapCamera.cx
			cameraMeta.cy = colmapCamera.cy
			
			// Map distortion parameters
			cameraMeta.k1 = colmapCamera.k1
			cameraMeta.k2 = colmapCamera.k2
			cameraMeta.k3 = colmapCamera.k3
			cameraMeta.p1 = colmapCamera.p1
			cameraMeta.p2 = colmapCamera.p2
			
			// Copy transform matrix (already in OpenGL convention)
			cameraMeta.camToWorld = colmapCamera.transformMatrix
			
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
			
			cameraMeta.width = polycamCamera.width
			cameraMeta.height = polycamCamera.height
			cameraMeta.fx = polycamCamera.fx
			cameraMeta.fy = polycamCamera.fy
			cameraMeta.cx = polycamCamera.cx
			cameraMeta.cy = polycamCamera.cy
			
			// Map distortion parameters (Polycam uses k1, k2, p1, p2)
			cameraMeta.k1 = polycamCamera.k1
			cameraMeta.k2 = polycamCamera.k2
			cameraMeta.k3 = 0  // Not provided by Polycam
			cameraMeta.p1 = polycamCamera.p1
			cameraMeta.p2 = polycamCamera.p2
			
			// Copy transform matrix (already in OpenGL convention)
			cameraMeta.camToWorld = polycamCamera.transformMatrix
			
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
		meta.width = w
		meta.height = h
		meta.fx = Float(fx ?? 0)
		meta.fy = Float(fy ?? 0)
		meta.cx = Float(cx ?? Double(w) / 2.0)
		meta.cy = Float(cy ?? Double(h) / 2.0)
		return meta
	}
}
