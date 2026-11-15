import SwiftUI
import MsplatCore
import AppKit
import QuartzCore



// MARK: - Vector helpers

func normalize(_ v: (Float, Float, Float)) -> (Float, Float, Float) {
    let len = sqrt(v.0*v.0 + v.1*v.1 + v.2*v.2)
    guard len > 1e-8 else { return (0, 1, 0) }
    return (v.0/len, v.1/len, v.2/len)
}

func cross(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> (Float, Float, Float) {
    (a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}

// MARK: - Orbit camera

/// Build a cam-to-world matrix (4x4 row-major, OpenGL: Y-up, Z-back)
/// looking from `eye` toward `target` with a given world-up hint.
func lookAtCamToWorld(eye: (Float, Float, Float), target: (Float, Float, Float),
                      up: (Float, Float, Float)) -> [Float] {
    // Forward = normalize(eye - target)  (camera looks along -Z in OpenGL)
    let f = normalize((eye.0 - target.0, eye.1 - target.1, eye.2 - target.2))
    // Right = normalize(up × forward)
    let r = normalize(cross(up, f))
    // True up = forward × right
    let u = cross(f, r)

    // Row-major cam-to-world: columns are right, up, forward; last column is eye position
    return [
        r.0, u.0, f.0, eye.0,
        r.1, u.1, f.1, eye.1,
        r.2, u.2, f.2, eye.2,
          0,   0,   0,     1,
    ]
}

struct OrbitParams {
    var lookAt: (Float, Float, Float)    // where the camera looks (scene focus)
    var eyeCenter: (Float, Float, Float) // center of the orbit circle (above lookAt)
    var radius: Float
    var up: (Float, Float, Float)        // scene up direction (from camera average)
    var tangent1: (Float, Float, Float)  // ground-plane basis vector 1
    var tangent2: (Float, Float, Float)  // ground-plane basis vector 2
}

/// Compute orbit parameters from dataset camera poses.
/// Derives the ground plane from the average camera up-vector (column 1 of camToWorld).
/// The orbit center is where cameras are looking, not where they are.
func computeOrbitParams(_ poses: [[Float]]) -> OrbitParams {
    let n = Float(poses.count)

    // Camera positions and up vectors
    var cx: Float = 0, cy: Float = 0, cz: Float = 0
    var ux: Float = 0, uy: Float = 0, uz: Float = 0

    for p in poses {
        cx += p[3]; cy += p[7]; cz += p[11]
        ux += p[1]; uy += p[5]; uz += p[9]
    }
    cx /= n; cy /= n; cz /= n
    let up = normalize((ux, uy, uz))

    // Orbit radius = average distance from centroid projected onto ground plane
    var totalR: Float = 0
    for p in poses {
        let dx = p[3] - cx, dy = p[7] - cy, dz = p[11] - cz
        let dot = dx*up.0 + dy*up.1 + dz*up.2
        let gx = dx - dot*up.0, gy = dy - dot*up.1, gz = dz - dot*up.2
        totalR += sqrt(gx*gx + gy*gy + gz*gz)
    }
    let radius = totalR / n * 1.3

    // Look-at target: below camera centroid (where the scene centerpiece is)
    let lookAt = (
        cx - up.0 * radius * 0.3,
        cy - up.1 * radius * 0.3,
        cz - up.2 * radius * 0.3
    )
    // Eye orbits slightly above the camera centroid so it looks down at the scene
    let eyeCenter = (
        cx - up.0 * radius * 0.07,
        cy - up.1 * radius * 0.07,
        cz - up.2 * radius * 0.07
    )

    // Build two tangent vectors spanning the ground plane
    let absX = abs(up.0), absY = abs(up.1), absZ = abs(up.2)
    let seed: (Float, Float, Float)
    if absX <= absY && absX <= absZ { seed = (1, 0, 0) }
    else if absY <= absZ { seed = (0, 1, 0) }
    else { seed = (0, 0, 1) }

    let t1 = normalize(cross(up, seed))
    let t2 = cross(up, t1)

    return OrbitParams(lookAt: lookAt, eyeCenter: eyeCenter, radius: radius,
                       up: up, tangent1: t1, tangent2: t2)
}

// MARK: - Engine

struct MemorySummary: Sendable {
	var totalMB: Double
	var efficiency: Double
	var modelMB: Double
	var datasetMB: Double
	var cpuImagesMB: Double
	var splatCount: Int
	
	var mbPerSplat: Double {
		guard splatCount > 0 else { return 0 }
		return modelMB / Double(splatCount)
	}
}

@MainActor
@Observable
final class Engine
{
	//	last Nth step render
	var image: CGImage?
	var splats : [PlySplatPoint] = []
	
	var iteration: Int = 0
	var totalIterations: Int = 10_000
	var splatCount: Int = 0
	var msPerStep: Float = 0
	var elapsedSecs : Float = 0
	var fps: Float = 0
	var phase : Phase
	var trainingTask : Task<Void,Never>!
	let metalContext : MetalContext
	
	// Memory tracking
	var memory = MemorySummary(totalMB: 0, efficiency: 0, modelMB: 0, datasetMB: 0, cpuImagesMB: 0, splatCount: 0)

	enum Phase
	{
		case error(Error)
		case loading(String,URL?)
		case training
		case orbiting
	}
	
	@MainActor deinit
	{
		trainingTask.cancel()
	}

	init(datasetPath: URL) throws
	{
		metalContext = try MetalContext()
		phase = .loading("Init", nil)
		trainingTask = Task.detached 
		{
			do
			{
				await MainActor.run
				{
					self.phase = .loading("DataSet",datasetPath)
				}
				
				let inputData = try await SplatDatasetLoader.load(from: datasetPath, colmapImagePath: nil)
				
				await MainActor.run
				{
					self.phase = .loading("Generating pyramids",datasetPath)
				}
				let splatDataset = try await SplatInputData(from: inputData, downscaleFactor: 1.0)
				{
					let newPhase = Phase.loading($0,$1)
					Task{ @MainActor in
						self.phase = newPhase
					}
				}
			
				await MainActor.run
				{
					self.phase = .training
				}
				try await self.performTraining(inputData: splatDataset, totalIterations: self.totalIterations)
			}
			catch
			{
				await MainActor.run
				{
					self.phase = .error(error)
				}
			}
		}
	}
	
	/// Error state initializer (when Metal context can't be created)
	init(error: Error) {
		// Use a dummy context that will never be accessed
		self.metalContext = try! MetalContext()
		self.phase = .error(error)
		self.trainingTask = Task { }
	}

	nonisolated private func performTraining(inputData:SplatInputData, totalIterations: Int) async throws
	{
		let inputData = inputData.autoScaleAndCenter()
		
		var config = MsplatConfig()
		config.iterations = Int32(totalIterations)
		config.numDownscales = 0
		config.bgColor = (0, 0, 0)
		
		let trainer = try SplatTrainer(context:metalContext, dataset: inputData, config: config)
		
		// Release inputData reference - trainer owns the dataset now
		_ = inputData

		// Phase 1: Training
		var batchSteps = 0
		let startTime = CACurrentMediaTime()
		var batchStart = startTime
		var renderCameraIndex = 0

		for i in 0..<config.iterations 
		{
			let stats = try await trainer.step()
			batchSteps += 1

			Task.detached {
				await MainActor.run 
				{
					self.iteration = Int(i)
				}
			}
			
			if i % 30 == 0 || i == config.iterations-1
			{
				let batchEnd = CACurrentMediaTime()
				let avgMs = Float((batchEnd - batchStart) / Double(batchSteps) * 1000.0)

				let splatRender = try await trainer.render(cameraIndex: renderCameraIndex % inputData.numCameras, format: .rgba)
				renderCameraIndex += 1
				
				let splats = trainer.GetSplats()
				
				//	this can be done on another thread
				do
				{
					let cgImage = try splatRender.makeCGImage(useFloatComponents: true)
					let iter = Int(stats.iteration)
					let count = Int(stats.splatCount)
					let elapsedSecs = batchEnd - startTime
					let memory = MemorySummary(
						totalMB: trainer.totalGPUMemoryUsageMB + trainer.datasetCPUMemoryUsageMB,
						efficiency: trainer.memoryEfficiency,
						modelMB: trainer.modelGPUMemoryUsageMB,
						datasetMB: trainer.datasetGPUMemoryUsageMB,
						cpuImagesMB: trainer.datasetCPUMemoryUsageMB,
						splatCount: count
					)
					Task.detached {
						await MainActor.run 
						{
							self.image = cgImage
							self.splats = splats
							self.iteration = iter
							self.splatCount = count
							self.msPerStep = avgMs
							self.elapsedSecs = Float(elapsedSecs)
							self.memory = memory
						}
					}
				}
				catch
				{
					await MainActor.run 
					{
						self.phase = .error(error)
					}
				}

				batchStart = CACurrentMediaTime()
				batchSteps = 0
			}
		}

		let finalCount = trainer.splatCount
		await MainActor.run {
			self.splatCount = finalCount
			self.phase = .orbiting
		}

		/*
		
		// Phase 2: Smooth circular orbit with zero-copy render
		let poses = inputData.cameras.map { $0.meta.camToWorld }
		let orbit = computeOrbitParams(poses)
		var frameCount = 0
		var fpsTimer = CACurrentMediaTime()

		// Query dimensions once to pre-allocate RGBA buffer
		var imgW: Int = 0, imgH: Int = 0
		let firstPose = lookAtCamToWorld(eye: orbit.eyeCenter, target: orbit.lookAt, up: orbit.up)
		(imgW, imgH) = try trainer.renderFromPose(camToWorld: firstPose, refCameraIndex: 0, into: nil)
		let rgbaBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: imgW * imgH * 4)
		for i in 0..<(imgW * imgH) { rgbaBuf[i * 4 + 3] = 255 }

		let colorSpace = CGColorSpaceCreateDeviceRGB()
		let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

		let orbitPeriod: Double = 7.5  // seconds per full revolution
		let orbitStart = CACurrentMediaTime()

		while true {
			let elapsed = CACurrentMediaTime() - orbitStart
			let angle = Float(elapsed / orbitPeriod) * 2.0 * .pi
			let cosA = cos(angle), sinA = sin(angle)
			let t1 = orbit.tangent1, t2 = orbit.tangent2
			let eye = (
				orbit.eyeCenter.0 + orbit.radius * (cosA * t1.0 + sinA * t2.0),
				orbit.eyeCenter.1 + orbit.radius * (cosA * t1.1 + sinA * t2.1),
				orbit.eyeCenter.2 + orbit.radius * (cosA * t1.2 + sinA * t2.2)
			)

			let pose = lookAtCamToWorld(eye: eye, target: orbit.lookAt, up: orbit.up)
			(imgW, imgH) = try trainer.renderFromPose(camToWorld: pose, refCameraIndex: 0, into: rgbaBuf)
			frameCount += 1

			var currentFps: Float = 0
			let now = CACurrentMediaTime()
			let dt = now - fpsTimer
			if dt >= 0.5 {
				currentFps = Float(frameCount) / Float(dt)
				frameCount = 0
				fpsTimer = now
			}

			// Only push to UI if main thread consumed previous frame
			if OSAtomicCompareAndSwap32(1, 0, &displayReady) {
				let provider = CGDataProvider(dataInfo: nil, data: rgbaBuf,
											   size: imgW * imgH * 4,
											   releaseData: { _, _, _ in })!
				let cgImg = CGImage(width: imgW, height: imgH,
									bitsPerComponent: 8, bitsPerPixel: 32,
									bytesPerRow: imgW * 4, space: colorSpace,
									bitmapInfo: bitmapInfo, provider: provider,
									decode: nil, shouldInterpolate: false,
									intent: .defaultIntent)!
				let img = NSImage(cgImage: cgImg, size: NSSize(width: imgW, height: imgH))
				let fpsVal = currentFps
				await MainActor.run {
					self.image = img
					if fpsVal > 0 { self.fps = fpsVal }
					OSAtomicCompareAndSwap32(0, 1, &displayReady)
				}
			}
		}
		 */
    }
}

extension CGImage
{
	var size : CGSize
	{
		return .init(width: width, height: height)
	}
}

struct ContentView: View
{
	@State var engine : Engine
	@State private var showDetailedMemory = true

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
		ZStack 
		{
			Color.black
			
			if let img = engine.image {
				Image(nsImage: NSImage(cgImage: img, size: img.size))
					.resizable()
					.aspectRatio(contentMode: .fit)
			}
			
			switch engine.phase
			{
				case .loading(let name,let url):	LoadingView(name: name, url:url)
				case .training, .orbiting:			TrainingView()
				case .error(let error):				ErrorView(error: error)
			}
		}
		.onKeyPress(.init("m")) {
			// Toggle detailed memory display
			showDetailedMemory.toggle()
			return .handled
		}
		.onKeyPress(.init("p")) {
			// Print memory stats to console
			print("GPU Memory: \(engine.memory.totalMB) MB (Model: \(engine.memory.modelMB) MB, Dataset: \(engine.memory.datasetMB) MB)")
			print("Efficiency: \(engine.memory.efficiency)%")
			return .handled
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
	
}

// MARK: - App entry

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct DemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    let datasetPath: URL

    init() {
        let args = CommandLine.arguments
		
		if let url = Bundle.module.url(forResource: "datasets/mipnerf360/garden", withExtension: nil) 
		{
			datasetPath = url  // Convert URL to path string
		}
		else
		{
			let path = args.count > 1 ? args[1] : "datasets/mipnerf360/garden"
			datasetPath = URL(filePath: path)
		}
    }

    var body: some Scene 
	{
        WindowGroup {
            ContentView(datasetPath: datasetPath)
                .frame(minWidth: 200, minHeight: 200)
        }
        .defaultSize(width: 1280, height: 720)
    }
}
