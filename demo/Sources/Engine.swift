import simd
import SwiftUI
import MsplatCore



func normalize(_ v: (Float, Float, Float)) -> (Float, Float, Float) {
	let len = sqrt(v.0*v.0 + v.1*v.1 + v.2*v.2)
	guard len > 1e-8 else { return (0, 1, 0) }
	return (v.0/len, v.1/len, v.2/len)
}

func cross(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> (Float, Float, Float) {
	(a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}



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
			
			if i == 100 || i % 200 == 0 || i == config.iterations-1
			{
				let batchEnd = CACurrentMediaTime()
				let avgMs = Float((batchEnd - batchStart) / Double(batchSteps) * 1000.0)
				
				let splatRender = try await trainer.render(cameraIndex: renderCameraIndex % inputData.numCameras, format: .rgba)
				renderCameraIndex += 1
				
				let splats = try trainer.GetSplats()
				
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
