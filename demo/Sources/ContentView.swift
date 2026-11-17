import SwiftUI
import MsplatCore
import QuartzCore
import SplatScene



struct ContentView: View
{
	@State var engine : Engine
	@State private var showDetailedMemory = true
	
	
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
				case .training, .orbiting:			TrainingView()
				case .error(let error):				ErrorView(error: error)
			}
		}
		.onChange(of: engine.splats)
		{
			old,new in
			OnEngineSplatsChanged(points:new)
		}
	}
	
	private var progressBar: some View {
		let progress = Double(engine.iteration) / Double(engine.totalIterations)
		return GeometryReader { geo in
			ZStack(alignment: .leading) {
				Rectangle()
					.fill(.white.opacity(0.15))
				Rectangle()
					.fill(
						LinearGradient(
							colors: [Color(red: 0.2, green: 0.5, blue: 1.0),
									 Color(red: 0.0, green: 0.9, blue: 0.7)],
							startPoint: .leading, endPoint: .trailing
						)
					)
					.frame(width: geo.size.width * progress)
					.animation(.linear(duration: 0.1), value: progress)
			}
		}
		.frame(height: 6)
	}
	
	private var statsBar: some View {
		HStack(spacing: 32) {
			switch engine.phase 
			{
				case .loading:
					Text("Loading...")
				case .error(let error):
					Text("Error: \(error.localizedDescription)")
				case .training:
					Text("step \(engine.iteration) / \(engine.totalIterations)")
					Text("\(fmtCount(engine.splatCount)) splats")
					Text(String(format: "%.1f ms/step", engine.msPerStep))
					Text("Elapsed \(Int(engine.elapsedSecs))secs")
					Text(String(format: "GPU: %.0f MB", engine.memory.totalMB))
						.foregroundStyle(memoryColor.opacity(0.9))
				case .orbiting:
					Text("\(fmtCount(engine.splatCount)) splats")
					Text(String(format: "%.0f fps", engine.fps))
			}
		}
		.font(.system(size: 18, weight: .semibold, design: .monospaced))
		.foregroundStyle(.white)
		.frame(maxWidth: .infinity)
		.padding(.vertical, 14)
		.background(.black.opacity(0.85))
	}
	
	private func fmtCount(_ n: Int) -> String {
		if n >= 1_000_000 { return String(format: "%.2fM", Double(n) / 1_000_000) }
		if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1_000) }
		return "\(n)"
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
			
			progressBar
			
			// Full-width stats bar
			statsBar
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
			.background {
				RoundedRectangle(cornerRadius: 8)
					.fill(.black.opacity(0.75))
					.overlay {
						RoundedRectangle(cornerRadius: 8)
							.strokeBorder(memoryColor.opacity(0.3), lineWidth: 1)
					}
			}
		
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
	
	@ViewBuilder func OrbitingView() -> some View
	{
		// Prominent FPS counter top-right during orbit
		TrainingView()
			.overlay
		{
			VStack {
				HStack {
					Spacer()
					Text(String(format: "%.0f FPS", engine.fps))
						.font(.system(size: 36, weight: .bold, design: .monospaced))
						.foregroundStyle(.white)
						.padding(.horizontal, 20)
						.padding(.vertical, 10)
						.background(.black.opacity(0.7))
						.cornerRadius(12)
						.padding(20)
				}
				Spacer()
			}
		}
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
			if let img = engine.trainerRenderedImage 
			{
				Image(nsImage: NSImage(cgImage: img, size: img.size))
					.resizable()
					.aspectRatio(contentMode: .fit)
					.frame(minWidth: 300,maxWidth: 500,maxHeight: .infinity)
			}
			if let img = engine.trainerRenderedImage2
			{
				Image(nsImage: NSImage(cgImage: img, size: img.size))
					.resizable()
					.aspectRatio(contentMode: .fit)
					.frame(minWidth: 300,maxWidth: 500,maxHeight: .infinity)
			}
			
			SplatSceneView(scene: scene, renderStats: $renderStats, renderer: .metal, assetManager:assetManager)
				.frame(minWidth: 300,maxWidth: .infinity,maxHeight: .infinity)
		}
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
