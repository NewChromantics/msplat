import Foundation
import ColmapData
import NerfStudioData
import PolycamData
import simd


extension PolycamCamera
{
	var intrinsics : CameraIntrinsics
	{
		.init(	fx: fx, 
				fy: fy, 
				cx: cx, 
				cy: cy, 
				width: width, 
				height: height
		)
	}
	
	var distortionParams : DistortionParams
	{
		.init(
			k1: k1,
			k2: k2,
			//k3: k3,
			//k4: k4,
			p1: p1,
			p2: p2
		)
		
	}
}

extension ColmapCameraWithTransform
{
	var intrinsics : CameraIntrinsics
	{
		.init(	fx: fx, 
				fy: fy, 
				cx: cx, 
				cy: cy, 
				width: width, 
				height: height
		)
	}
	
	var distortionParams : DistortionParams
	{
		.init(
			k1: k1,
			k2: k2,
			k3: k3,
			/*k4: k4,*/// k4 is ignored - not supported by our distortion model
			p1: p1,
			p2: p2
		)
		
	}
}

extension NerfStudioCamera
{
	var intrinsics : CameraIntrinsics
	{
		.init(	fx: fx, 
				fy: fy, 
				cx: cx, 
				cy: cy, 
				width: width, 
				height: height
		)
	}
	
	var distortionParams : DistortionParams
	{
		// Map distortion parameters (NerfStudio uses k1-k4, p1-p2; we use k1-k3, p1-p2)
		.init(
			k1: k1,
			k2: k2,
			k3: k3,
			/*k4: k4,*/// k4 is ignored - not supported by our distortion model
			p1: p1,
			p2: p2
			)
		
	}
}


/// Convert a 16-element Float array (row-major) to simd_float4x4 (column-major).
/// This handles the conversion from the typical row-major format used in JSON files
/// to the column-major format used by SIMD types.
@available(*, deprecated, renamed: "simd_float4x4(rowMajor:)", message: "use constructor")
private func arrayToMatrix4x4(_ array: [Float]) -> simd_float4x4 
{
	return simd_float4x4(rowMajor: array)
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
		for nerfCamera in nerfDataset.cameras 
		{
			let imageSource = ImageSource.url(nerfDataset.baseDirectory.appendingPathComponent(nerfCamera.imagePath))
			let camToWorld = simd_float4x4(rowMajor: nerfCamera.transformMatrix)
			let cameraMeta = CameraMeta(imageSource:imageSource, intrinsics: nerfCamera.intrinsics, distortion: nerfCamera.distortionParams, camToWorld:camToWorld)
			
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
			let imageSource = ImageSource.url(colmapCamera.imagePath)
			let camToWorld = simd_float4x4(colMajor: colmapCamera.transformMatrix)
			let cameraMeta = CameraMeta(imageSource:imageSource, intrinsics: colmapCamera.intrinsics, distortion: colmapCamera.distortionParams, camToWorld: camToWorld)
			
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
		for polycamCamera in polycamDataset.cameras 
		{
			let imageSource = ImageSource.url(polycamDataset.baseDirectory.appendingPathComponent(polycamCamera.imagePath))
			let camToWorld = simd_float4x4(rowMajor:polycamCamera.transformMatrix)
			let cameraMeta = CameraMeta(imageSource:imageSource, intrinsics: polycamCamera.intrinsics, distortion: polycamCamera.distortionParams, camToWorld: camToWorld)
			
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


