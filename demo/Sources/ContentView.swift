import SwiftUI
import MsplatCore
import QuartzCore
import SplatScene
import simd




struct SplatArrayStatsView : View 
{
	var splats : [PlySplatPoint] = []
	
	var body: some View 
	{
		VStack
		{
			Text("\(splats.count) splats")
		}
	}
}



struct ContentView: View
{
	@State var engine : Engine
	@State private var showDetailedMemory = true
	
	@State var splatSceneCameraState = SplatScene.CameraState()
	@State var scene : SplatScene.Scene = GetInitialScene()
	@State var assetManager = AssetManager()
	@State var renderStats = RenderStats()
	static let liveSplatAssetUrl = URL(string:"livesplat")!
	
	init(datasetPath:URL)
	{
		do {
			engine = try Engine(datasetPath: datasetPath)
		} catch {
			// If context creation fails, create a dummy engine in error state
			engine = Engine(error: error)
		}
	}
	
	var body: some View 
	{
		RenderView()
			.background(.black)
			.overlay
		{
			switch engine.phase
			{
				case .loading(let name,let url):	LoadingView(name: name, url:url)
				case .training, .finished:			TrainingView()
				case .error(let error):				ErrorView(error: error)
			}
		}
		.onChange(of: engine.splats)
		{
			old,new in
			OnEngineSplatsChanged(points:new)
		}
	}
	
	
	@ViewBuilder func LoadingView(name:String,url:URL?) -> some View
	{
		let filename = url.map{ $0.lastPathComponent } ?? ""
		VStack 
		{
			ProgressView()
				.scaleEffect(1.5)
				.tint(.white)
			Text("Loading \(filename) \(name)...")
				.foregroundStyle(.white)
				.padding(.top, 8)
		}
	}
	
	@ViewBuilder func ErrorView(error:Error) -> some View
	{
		Text("Error: \(error.localizedDescription)")
			.padding(10)
			.background(.red)
			.foregroundStyle(.white)
	}
	
	@ViewBuilder func TrainingView() -> some View
	{
		VStack(spacing: 0) 
		{
			Spacer()
		}
		.overlay(alignment: .topTrailing) 
		{
			// Memory usage overlay in top-right corner
			MemoryOverlay()
				.padding(16)
		}
	}
	
	
	@ViewBuilder func MemoryStats() -> some View
	{
		VStack(alignment: .trailing, spacing: 6) 
		{
			Text("GPU MEMORY")
				.font(.system(size: 10, weight: .bold, design: .monospaced))
				.foregroundStyle(.white.opacity(0.6))
			
			Text(String(format: "%.1f MB", engine.memory.totalMB))
				.font(.system(size: 24, weight: .bold, design: .monospaced))
				.foregroundStyle(memoryColor)
			
			HStack(spacing: 4) {
				Circle()
					.fill(efficiencyColor)
					.frame(width: 6, height: 6)
				Text(String(format: "%.0f%% efficient", engine.memory.efficiency))
					.font(.system(size: 11, weight: .medium, design: .monospaced))
					.foregroundStyle(.white.opacity(0.8))
			}
			
			if showDetailedMemory {
				Divider()
					.frame(maxWidth:100)
					.background(.white.opacity(0.3))
					.padding(.vertical, 4)
				
				VStack(alignment: .trailing, spacing: 2) {
					Text(memoryBreakdownText)
						.font(.system(size: 9, design: .monospaced))
						.foregroundStyle(.white.opacity(0.7))
				}
			}
			
			// Help text
			Text("M: toggle • P: print")
				.font(.system(size: 8, design: .monospaced))
				.foregroundStyle(.white.opacity(0.4))
				.padding(.top, 4)
		}
	}
	
	@ViewBuilder func MemoryOverlay() -> some View 
	{
		MemoryStats()
			.padding(12)
			.darkBackground()
	}
	
	private var memoryColor: Color {
		if engine.memory.totalMB > 4000 {
			return .red
		} else if engine.memory.totalMB > 2000 {
			return .orange
		} else {
			return Color(red: 0.0, green: 0.9, blue: 0.7)
		}
	}
	
	private var efficiencyColor: Color {
		if engine.memory.efficiency < 15 {
			return .red
		} else if engine.memory.efficiency < 25 {
			return .orange
		} else {
			return .green
		}
	}
	
	private var memoryBreakdownText: String
	{
		let kbPerSplat = engine.memory.mbPerSplat * 1024.0
		return String(format: "Model: %.1f MB\nGPU Cache: %.1f MB\nCPU Images: %.1f MB\n%.1f KB/splat", 
					  engine.memory.modelMB, engine.memory.datasetMB, engine.memory.cpuImagesMB, kbPerSplat)
	}
	
	
	static func GetInitialScene() -> SplatScene.Scene
	{
		let actors = [
			SceneActorUid() : SceneActor(position: .zero, assetUrl: liveSplatAssetUrl )
			]
		return .init(actors: actors)
	}
	
	@ViewBuilder func RenderView() -> some View
	{
		HSplitView
		{
			EngineStateView(engine: engine, OnClickedSetSplatPose:SetSplatPose)
				.frame(maxWidth: .infinity,maxHeight: .infinity)
			
			SplatSceneView(scene: scene, renderStats: $renderStats, renderer: .metal, assetManager:assetManager, cameraState: $splatSceneCameraState)
				.frame(minWidth: 300,maxWidth: .infinity,maxHeight: .infinity)
		}
	}
	

		
	func SetSplatPose(cameraMeta:CameraMeta)
	{
		let transform = cameraMeta.camToWorld.transpose
		
		// Extract camera position (translation) from column 3
		let position = SIMD3<Float>(
			transform.columns.3.x,
			transform.columns.3.y,
			transform.columns.3.z
		)
		
		// Extract rotation from the transform matrix
		// In OpenGL camera-to-world convention:
		// Column 0 = right vector
		// Column 1 = up vector  
		// Column 2 = forward (camera looks along -Z, so forward is +Z in world)
		let forward = SIMD3<Float>(
			transform.columns.2.x,
			transform.columns.2.y,
			transform.columns.2.z
		)
		let up = SIMD3<Float>(
			transform.columns.1.x,
			transform.columns.1.y,
			transform.columns.1.z
		)
		
		// Calculate yaw (rotation around Y axis) from forward vector
		// atan2(x, z) gives angle in XZ plane
		let yaw = atan2(forward.x, forward.z)
		
		// Calculate pitch (rotation around X axis)
		// Project forward vector length in XZ plane, then get angle with Y component
		let forwardXZ = sqrt(forward.x * forward.x + forward.z * forward.z)
		let pitch = atan2(-forward.y, forwardXZ)
		
		// Calculate distance from origin (or from a target point if you have one)
		let distance = length(position)
		
		// Update the camera state
		// Note: You may need to adjust these assignments based on the actual
		// properties available in SplatScene.CameraState
		var newCameraState = SplatScene.CameraState()
		
		// If CameraState has these properties, uncomment and adjust as needed:
		//newCameraState.position = position
		newCameraState.yaw = Angle(radians:Double(yaw))
		newCameraState.pitch = Angle(radians: Double(pitch))
		newCameraState.distance = distance
		newCameraState.lookAtPosition = SIMD3<Float>(0, 0, 0) // or calculate from forward vector
		
		// For now, since we don't know the exact structure of CameraState,
		// we can try to construct it with the transform directly if supported:
		self.splatSceneCameraState = newCameraState
		//self.splatSceneCameraState.overridingLocalToWorldTransform = cameraMeta.camToWorld.transpose
		
		// Debug output to help verify the values
		print("Camera Pose Copied:")
		print("  Position: \(newCameraState.position) (expected \(position))")
		print("  Yaw: \(yaw * 180 / .pi)°")
		print("  Pitch: \(pitch * 180 / .pi)°")
		print("  Distance: \(distance)")
	}
	
	@State var loadCount = 0
	func OnEngineSplatsChanged(points:[PlySplatPoint])
	{
		//	asset data is flat interleaved
		let floats : [Float] = points.flatMap
		{
			plySplatPoint in
			// RealityKit's GaussianSplatResource expects specific memory layout:
			// position(3) scale(3) rotation(4) opacity(1) sphericalHarmonics(5+)
			// Even for degree .zero (3 SH coeffs), we need to pad to at least 5 floats
			// based on Metal validation requiring 20 bytes (5 floats) for the SH argument
			[
				plySplatPoint.position.x,
				plySplatPoint.position.y,
				plySplatPoint.position.z,
				plySplatPoint.scaleLog.x,
				plySplatPoint.scaleLog.y,
				plySplatPoint.scaleLog.z,
				plySplatPoint.quaternionWxyz[0],
				plySplatPoint.quaternionWxyz[1],
				plySplatPoint.quaternionWxyz[2],
				plySplatPoint.quaternionWxyz[3],
				plySplatPoint.opacityLogit,
				
				plySplatPoint.shDc.x,
				plySplatPoint.shDc.y,
				plySplatPoint.shDc.z,
				//0,	// SH padding
				//0,	// SH padding (16 floats total = 64 bytes, 16-byte aligned)
			]
		}
		
		// 16 floats per splat = 64 bytes (16-byte aligned)
		// SH data is 3 floats but padded with 2 extra floats for Metal validation
		let splatData = SplatAsset.SplatData(floatsPerSplat: 14, sphericalHarmonicsFloats: 3, interleaved: floats)
		//	make new asset
		let asset = SplatAsset(splatData: splatData)
		
		assetManager.InsertAsset(url: Self.liveSplatAssetUrl, asset: asset)
	}
}
