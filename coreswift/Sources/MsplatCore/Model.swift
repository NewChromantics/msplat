// Port of model.hpp / model.cpp: gaussian parameters, Adam state in oversized
// backing buffers, schedules, densification trigger, render / training iteration,
// PLY/splat export and checkpointing.

import Foundation
import simd


extension CameraMeta
{
	//	featuresRestSizeSecondLast is
	//		featuresRest.size(-2)
	//	so its the length of... why penultimate?...
	public func GetCameraRenderParams(metalContext:MetalContext,cameraIntrinsics:CameraIntrinsics,shDegrees:Int,shDegreesToUse:Int) throws -> CameraRenderParams 
	{
		let fovX = intrinsics.fovX
		let fovY = intrinsics.fovY
		let d = self.camToWorld
		
		// Extract rotation (flip Y and Z for OpenGL to view space conversion)
		var R = [[Float]](repeating: [0, 0, 0], count: 3)
		var T: [Float] = [0, 0, 0]
		for i in 0..<3 {
			R[i][0] = d[i][0]
			R[i][1] = -d[i][1]
			R[i][2] = -d[i][2]
			T[i] = d[i][3]
		}
		
		// Transpose R to get inverse (rotation matrices are orthonormal)
		var Rinv = [[Float]](repeating: [0, 0, 0], count: 3)
		for i in 0..<3 {
			for j in 0..<3 {
				Rinv[i][j] = R[j][i]
			}
		}
		
		// Compute inverse translation: Tinv = -Rinv * T
		var Tinv: [Float] = [0, 0, 0]
		for i in 0..<3 {
			Tinv[i] = -(Rinv[i][0] * T[0] + Rinv[i][1] * T[1] + Rinv[i][2] * T[2])
		}
		
		// Build view matrix (world-to-camera)
		let viewMatrix: [Float] = [
			Rinv[0][0], Rinv[0][1], Rinv[0][2], Tinv[0],
			Rinv[1][0], Rinv[1][1], Rinv[1][2], Tinv[1],
			Rinv[2][0], Rinv[2][1], Rinv[2][2], Tinv[2],
			0, 0, 0, 1,
		]
		
		// Build projection matrix
		let tP = 0.001 * tanf(0.5 * fovY)
		let rP = 0.001 * tanf(0.5 * fovX)
		let zf: Float = 1000.0
		let zn: Float = 0.001
		
		let projectionMatrix: [Float] = [
			zn / rP, 0, 0, 0,
			0, zn / tP, 0, 0,
			0, 0, (zf + zn) / (zf - zn), -zf * zn / (zf - zn),
			0, 0, 1, 0,
		]
		
		// Compute projection * view matrix
		var pvm = [Float](repeating: 0, count: 16)
		for i in 0..<4 {
			for j in 0..<4 {
				for k in 0..<4 {
					pvm[i * 4 + j] += projectionMatrix[i * 4 + k] * viewMatrix[k * 4 + j]
				}
			}
		}
		
		
		let tileBounds = ((intrinsics.width + BLOCK_X - 1) / BLOCK_X, (intrinsics.height + BLOCK_Y - 1) / BLOCK_Y, 1)

		
		var cameraRenderParams = CameraRenderParams(degrees:shDegrees,
													degreesToUse: shDegreesToUse, 
													tileBounds:tileBounds, 
													intrinsics: cameraIntrinsics, 
													camToWorld: self.camToWorld)
		cameraRenderParams.viewMat = try metalContext.makeEmpty([4, 4])
		cameraRenderParams.viewMat.copyIn(viewMatrix)
		cameraRenderParams.projectionMatrix = simd_float4x4(rowMajor:projectionMatrix)
		cameraRenderParams.projViewMat = try metalContext.makeEmpty([4, 4])
		cameraRenderParams.projViewMat.copyIn(pvm)
		cameraRenderParams.fovX = fovX
		cameraRenderParams.fovY = fovY
		return cameraRenderParams
	}
}


/// Little-endian binary reader over Data.
struct BinaryReader 
{
	public struct BinaryReaderError: LocalizedError 
	{
		public var errorDescription: String?
		
		public init(_ description: String) 
		{
			self.errorDescription = description
		}
	}
	

	
	let data: Data
	var offset = 0
	
	init(_ data: Data) { self.data = data }
	
	var remaining: Int { data.count - offset }
	
	mutating func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
		let n = MemoryLayout<T>.size
		guard remaining >= n else { throw BinaryReaderError("Unexpected end of file") }
		let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
		offset += n
		return T(littleEndian: v)
	}
	
	mutating func readF64() throws -> Double { Double(bitPattern: try read(UInt64.self)) }
	mutating func readF32() throws -> Float { Float(bitPattern: try read(UInt32.self)) }
	
	mutating func readCString() throws -> String {
		var bytes: [UInt8] = []
		while remaining > 0 {
			let b = try read(UInt8.self)
			if b == 0 { break }
			bytes.append(b)
		}
		return String(decoding: bytes, as: UTF8.self)
	}
	
	mutating func skip(_ n: Int) throws {
		guard n >= 0, remaining >= n else { throw BinaryReaderError("Unexpected end of file") }
		offset += n
	}
}

private let C0 = 0.28209479177387814
private let BLOCK_X = 16
private let BLOCK_Y = 16

func numShBases(_ degree: Int) -> Int {
    switch degree {
    case 0: return 1
    case 1: return 4
    case 2: return 9
    case 3: return 16
    default: return 25
    }
}

// MARK: - SIMD Accessors for Float Arrays

extension UnsafeMutablePointer where Pointee == Float {
    /// Access 3 consecutive floats as simd_float3
    subscript(float3 index: Int) -> simd_float3 {
        get {
            let offset = index * 3
            return simd_float3(self[offset], self[offset + 1], self[offset + 2])
        }
        set {
            let offset = index * 3
            self[offset] = newValue.x
            self[offset + 1] = newValue.y
            self[offset + 2] = newValue.z
        }
    }
    
    /// Access 4 consecutive floats as simd_float4
    subscript(float4 index: Int) -> simd_float4 {
        get {
            let offset = index * 4
            return simd_float4(self[offset], self[offset + 1], self[offset + 2], self[offset + 3])
        }
        set {
            let offset = index * 4
            self[offset] = newValue.x
            self[offset + 1] = newValue.y
            self[offset + 2] = newValue.z
            self[offset + 3] = newValue.w
        }
    }
}


/// Detailed breakdown of GPU memory usage by component
public struct GPUMemoryBreakdown {
	public var gaussianParams: Int = 0      // means, scales, quats, features, opacities
	public var gaussianBuffers: Int = 0      // backing buffers (*Buf)
	public var adamState: Int = 0            // adamExpAvg, adamExpAvgSq
	public var adamBuffers: Int = 0          // adamExpAvgBuf, adamExpAvgSqBuf
	public var densificationScratch: Int = 0 // all densify* tensors
	public var trainingState: Int = 0        // xysGradNorm, visCounts, max2DSize, radii
	public var other: Int = 0                // window2d
	
	public var total: Int {
		gaussianParams + gaussianBuffers + adamState + adamBuffers + 
		densificationScratch + trainingState + other
	}
	
	public var totalMB: Double { Double(total) / (1024.0 * 1024.0) }
	
	public var description: String {
			"""
			GPU Memory Breakdown:
			  Gaussian Params:      \(formatBytes(gaussianParams))
			  Gaussian Buffers:     \(formatBytes(gaussianBuffers))
			  Adam State:           \(formatBytes(adamState))
			  Adam Buffers:         \(formatBytes(adamBuffers))
			  Densification Scratch:\(formatBytes(densificationScratch))
			  Training State:       \(formatBytes(trainingState))
			  Other:                \(formatBytes(other))
			  ─────────────────────────────────
			  Total:                \(formatBytes(total)) (\(String(format: "%.1f", totalMB)) MB)
			"""
	}
	
	private func formatBytes(_ bytes: Int) -> String {
		if bytes >= 1024 * 1024 * 1024 {
			return String(format: "%6.2f GB", Double(bytes) / (1024.0 * 1024.0 * 1024.0))
		} else if bytes >= 1024 * 1024 {
			return String(format: "%6.2f MB", Double(bytes) / (1024.0 * 1024.0))
		} else if bytes >= 1024 {
			return String(format: "%6.2f KB", Double(bytes) / 1024.0)
		} else {
			return String(format: "%6d B ", bytes)
		}
	}
}

final class Model
{
    static let nAdamGroups = 6

    //	Metal context for GPU operations
    let metalContext: MetalContext

    //	Active views (first numActive rows of the *_buf backing buffers)
	var numPoints : Int		{	means.size(0)	}
    var means = MTensor()	//	positions
	var scales = MTensor()	//	log scale
	var quats = MTensor()	//	wxyz
    var featuresDc = MTensor()
	var featuresRest = MTensor()
	var opacities = MTensor()	//	log^sigmoid

    var adamExpAvg = [MTensor](repeating: MTensor(), count: nAdamGroups)
    var adamExpAvgSq = [MTensor](repeating: MTensor(), count: nAdamGroups)
    var adamStepCount = 0
    var adamLr = [Float](repeating: 0, count: nAdamGroups)
    let adamBeta1: Float = 0.9, adamBeta2: Float = 0.999, adamEps: Float = 1e-8
    var meansLrInit: Float = 0, meansLrFinal: Float = 0

    // Backing buffers with headroom for densification
    var meansBuf = MTensor(), scalesBuf = MTensor(), quatsBuf = MTensor()
    var featuresDcBuf = MTensor(), featuresRestBuf = MTensor(), opacitiesBuf = MTensor()
    var adamExpAvgBuf = [MTensor](repeating: MTensor(), count: nAdamGroups)
    var adamExpAvgSqBuf = [MTensor](repeating: MTensor(), count: nAdamGroups)
    var numActive = 0, bufCapacity = 0

    // Densification scratch
    var densifySplitFlag = MTensor(), densifyDupFlag = MTensor()
    var densifySplitPrefix = MTensor(), densifyDupPrefix = MTensor()
    var densifyKeepFlag = MTensor(), densifyKeepPrefix = MTensor()
    var densifyBlockTotals = MTensor()
    var densifyCompactScratch = MTensor()
    var densifyRandomSamples = MTensor()

    var radii = MTensor()
	//	these should be passed around after step
    var lastHeight = 0
	var lastWidth = 0

    var xysGradNorm = MTensor(), visCounts = MTensor(), max2DSize = MTensor()

    var backgroundColor: simd_float3 = simd_float3(0, 0, 0)
    var window2d = MTensor()  // SSIM window (11, 11)

    let numCameras: Int
    let numDownscales: Int
    let resolutionSchedule: Int
    let shDegree: Int
    let shDegreeInterval: Int
    let refineEvery: Int
    let warmupLength: Int
    let resetAlphaEvery: Int
    let stopSplitAt: Int
    let densifyGradThresh: Float
    let densifySizeThresh: Float
    let stopScreenSizeAt: Int
    let splitScreenSize: Float
    let maxSteps: Int
    let keepCrs: Bool

    var scale: Float
    var translation: (Float, Float, Float)

    deinit {
        releaseOptimizers()
        // Release remaining GPU tensors
        means.reset()
		scales.reset()
		quats.reset()
        featuresDc.reset()
		featuresRest.reset()
		opacities.reset()
		window2d.reset()
        radii.reset()
		xysGradNorm.reset()
		visCounts.reset()
		max2DSize.reset()
    }

	init(metalContext: MetalContext, inputData: SplatInputData, numCameras: Int, numDownscales: Int, resolutionSchedule: Int,
         shDegree: Int, shDegreeInterval: Int, refineEvery: Int, warmupLength: Int,
         resetAlphaEvery: Int, densifyGradThresh: Float, densifySizeThresh: Float,
         stopScreenSizeAt: Int, splitScreenSize: Float, maxSteps: Int, keepCrs: Bool,
         bgColor: (Float, Float, Float)? = nil) throws {
        self.metalContext = metalContext
        self.numCameras = numCameras
        self.numDownscales = numDownscales
        self.resolutionSchedule = resolutionSchedule
        self.shDegree = shDegree
        self.shDegreeInterval = shDegreeInterval
        self.refineEvery = refineEvery
        self.warmupLength = warmupLength
        self.resetAlphaEvery = resetAlphaEvery
        self.stopSplitAt = maxSteps / 2
        self.densifyGradThresh = densifyGradThresh
        self.densifySizeThresh = densifySizeThresh
        self.stopScreenSizeAt = stopScreenSizeAt
        self.splitScreenSize = splitScreenSize
        self.maxSteps = maxSteps
        self.keepCrs = keepCrs
        scale = inputData.scale
        translation = inputData.translation

        let n = inputData.points.count

        // Means: xyz straight to GPU
		means = try metalContext.makeEmpty([n, 3])
        means.copyIn(inputData.points.xyz)

        // Scales: log of mean KNN distance, repeated 3x
        do {
            let sc = knnScales(xyz: inputData.points.xyz, count: n)
			scales = try metalContext.makeEmpty([n, 3])
            let sp = scales.floats()
            for i in 0..<n {
                let v = logf(sc[i])
                sp[i * 3] = v; sp[i * 3 + 1] = v; sp[i * 3 + 2] = v
            }
        }

        // Random unit quaternions (same mt19937(42) stream as C++)
        do {
            var rng = MT19937(seed: 42)
            let dist = UniformRealFloat(0, 1)
			quats = try metalContext.makeEmpty([n, 4])
            let qp = quats.floats()
            for i in 0..<n {
                let u = dist.sample(&rng), v = dist.sample(&rng), w = dist.sample(&rng)
                // C++: std::sqrt(float) * std::sin(double) → double product, stored as float
                let su = Double((1 - u).squareRoot()), sv = Double(u.squareRoot())
                qp[i * 4 + 0] = Float(su * sin(2 * Double.pi * Double(v)))
                qp[i * 4 + 1] = Float(su * cos(2 * Double.pi * Double(v)))
                qp[i * 4 + 2] = Float(sv * sin(2 * Double.pi * Double(w)))
                qp[i * 4 + 3] = Float(sv * cos(2 * Double.pi * Double(w)))
            }
        }

        // SH: f_dc = rgb2sh(rgb), f_rest = 0
        let dimSh = numShBases(shDegree)
		featuresDc = try metalContext.makeEmpty([n, 3])
        let dp = featuresDc.floats()
        for i in 0..<(n * 3) {
            dp[i] = Float((Double(inputData.points.rgb[i]) / 255.0 - 0.5) / C0)
        }
		featuresRest = try metalContext.makeZeros([n, dimSh - 1, 3])

        // Opacities: logit(0.1)
		opacities = try metalContext.makeEmpty([n, 1])
        let logit01 = logf(Float(0.1) / Float(0.9))
        let op = opacities.floats()
        for i in 0..<n { op[i] = logit01 }

        // Background (default magenta: makes under-reconstructed regions obvious)
        let bg = bgColor ?? (0.6130, 0.0101, 0.3984)
        backgroundColor = simd_float3(bg.0, bg.1, bg.2)

        setupOptimizers()
    }

    private static let lrInit: [Float] = [0.00016, 0.005, 0.001, 0.0025, 0.000125, 0.05]

    private var paramViews: [MTensor] { [means, scales, quats, featuresDc, featuresRest, opacities] }

    /// Oversized zeroed buffer with `param`'s row shape, prefilled with its data.
    private func allocBuf(_ param: MTensor) throws -> MTensor {
        var shape = param.shape
        shape[0] = bufCapacity
        let buf = try metalContext.makeZeros(shape)
        memcpy(buf.rawPointer, param.rawPointer, param.nbytes)
        return buf
    }

    private func allocDensifyScratch(capacity: Int, frStride: Int) throws {
		densifySplitFlag = try metalContext.makeZeros([capacity], .int32)
		densifyDupFlag = try metalContext.makeZeros([capacity], .int32)
		densifySplitPrefix = try metalContext.makeZeros([capacity], .int32)
		densifyDupPrefix = try metalContext.makeZeros([capacity], .int32)
		densifyKeepFlag = try metalContext.makeZeros([capacity], .int32)
		densifyKeepPrefix = try metalContext.makeZeros([capacity], .int32)
		densifyBlockTotals = try metalContext.makeZeros([(capacity + 1023) / 1024], .int32)
		densifyCompactScratch = try metalContext.makeZeros([capacity * frStride], .float32)
		densifyRandomSamples = try metalContext.makeZeros([capacity, 3], .float32)
    }

    func setupOptimizers() {
        releaseOptimizers()

        numActive = means.size(0)
        bufCapacity = numActive * 4
        meansBuf = try! allocBuf(means)
        scalesBuf = try! allocBuf(scales)
        quatsBuf = try! allocBuf(quats)
        featuresDcBuf = try! allocBuf(featuresDc)
        featuresRestBuf = try! allocBuf(featuresRest)
        opacitiesBuf = try! allocBuf(opacities)

        let params = paramViews
        for g in 0..<Model.nAdamGroups {
            var shape = params[g].shape
            shape[0] = bufCapacity
            adamExpAvgBuf[g] = try! metalContext.makeZeros(shape)
            adamExpAvgSqBuf[g] = try! metalContext.makeZeros(shape)
            adamLr[g] = Model.lrInit[g]
        }
        adamStepCount = 0
        meansLrInit = 0.00016
        meansLrFinal = 0.0000016

        try! allocDensifyScratch(capacity: bufCapacity, frStride: featuresRest.numel / featuresRest.size(0))
        refreshViews()
    }

    func releaseOptimizers() {
        // Ensure all GPU operations complete before releasing buffers
        //metalContext.gpuSync()
        
        for g in 0..<Model.nAdamGroups {
            adamExpAvg[g].reset(); adamExpAvgSq[g].reset()
            adamExpAvgBuf[g].reset(); adamExpAvgSqBuf[g].reset()
        }
        meansBuf.reset(); scalesBuf.reset(); quatsBuf.reset()
        featuresDcBuf.reset(); featuresRestBuf.reset(); opacitiesBuf.reset()
        densifySplitFlag.reset(); densifyDupFlag.reset()
        densifySplitPrefix.reset(); densifyDupPrefix.reset()
        densifyKeepFlag.reset(); densifyKeepPrefix.reset()
        densifyBlockTotals.reset(); densifyCompactScratch.reset(); densifyRandomSamples.reset()
    }

    func schedulersStep(_ step: Int) {
        let t = min(max(Float(step) / Float(maxSteps), 0), 1)
        adamLr[0] = expf(logf(meansLrInit) * (1 - t) + logf(meansLrFinal) * t)
    }

    func refreshViews() {
        means = meansBuf.view(numActive)
        scales = scalesBuf.view(numActive)
        quats = quatsBuf.view(numActive)
        featuresDc = featuresDcBuf.view(numActive)
        featuresRest = featuresRestBuf.view(numActive)
        opacities = opacitiesBuf.view(numActive)
        for g in 0..<Model.nAdamGroups {
            adamExpAvg[g] = adamExpAvgBuf[g].view(numActive)
            adamExpAvgSq[g] = adamExpAvgSqBuf[g].view(numActive)
        }
    }

    func ensureCapacity(_ needed: Int) {
        if needed <= bufCapacity { return }
        let newCap = max(needed, bufCapacity * 2)

        func grow(_ buf: MTensor) -> MTensor {
            var shape = buf.shape
            shape[0] = newCap
            let nb = try! metalContext.makeZeros(shape)
            memcpy(nb.rawPointer, buf.rawPointer, numActive * buf.stride0 * MemoryLayout<Float>.size)
            return nb
        }
        
        // Release old densification scratch buffers BEFORE allocating new ones
        densifySplitFlag.reset(); densifyDupFlag.reset()
        densifySplitPrefix.reset(); densifyDupPrefix.reset()
        densifyKeepFlag.reset(); densifyKeepPrefix.reset()
        densifyBlockTotals.reset(); densifyCompactScratch.reset(); densifyRandomSamples.reset()
        
        meansBuf = grow(meansBuf); scalesBuf = grow(scalesBuf); quatsBuf = grow(quatsBuf)
        featuresDcBuf = grow(featuresDcBuf); featuresRestBuf = grow(featuresRestBuf)
        opacitiesBuf = grow(opacitiesBuf)
        for g in 0..<Model.nAdamGroups {
            adamExpAvgBuf[g] = grow(adamExpAvgBuf[g])
            adamExpAvgSqBuf[g] = grow(adamExpAvgSqBuf[g])
        }
        try! allocDensifyScratch(capacity: newCap, frStride: featuresRestBuf.stride0)
        bufCapacity = newCap
        refreshViews()
    }

    func getDownscaleFactor(_ step: Int) -> Int {
        let remaining = numDownscales - step / resolutionSchedule
        return 1 << max(remaining, 0)
    }

    func afterTrain(_ step: Int) async throws
	{
        guard radii.defined else { return }
        guard step % refineEvery == 0 && step > warmupLength else { return }

        let resetInterval = resetAlphaEvery * refineEvery
        let doDensification = step < stopSplitAt && step % resetInterval > numCameras + refineEvery

        if doDensification {
            let numPointsBefore = numActive
            ensureCapacity(3 * numActive)  // worst case: every gaussian splits

            // Normal samples for splits (CPU, shared memory), same stream as C++
            do {
                var rng = MT19937(seed: UInt32(truncatingIfNeeded: step))
                var dist = NormalFloat(mean: 0, stddev: 1)
                let p = densifyRandomSamples.floats()
				let pointCount = 2 * numActive * 3
				print("Calulating \(pointCount) random number samples")
                for i in 0..<pointCount
				{
					p[i] = dist.sample(&rng) 
				}
            }

            let checkHuge = step > refineEvery * resetAlphaEvery
			let denisifyParams = DensifyParams(
				n: numActive, bufCapacity: bufCapacity,
				gradThresh: densifyGradThresh, sizeThresh: densifySizeThresh,
				screenThresh: splitScreenSize,
				checkScreen: step < stopScreenSizeAt ? 1 : 0,
				cullAlphaThresh: 0.1, cullScaleThresh: 0.5, cullScreenSize: 0.15,
				checkHuge: checkHuge ? 1 : 0,
				xysGradNorm: xysGradNorm, visCounts: visCounts, max2dSize: max2DSize,
				halfMaxDim: 0.5 * Float(max(lastWidth, lastHeight)),
				meansBuf: meansBuf, scalesBuf: scalesBuf, quatsBuf: quatsBuf,
				featuresDcBuf: featuresDcBuf, featuresRestBuf: featuresRestBuf,
				opacitiesBuf: opacitiesBuf, frStride: featuresRestBuf.stride0,
				adamExpAvgBuf: adamExpAvgBuf, adamExpAvgSqBuf: adamExpAvgSqBuf,
				splitFlag: densifySplitFlag, dupFlag: densifyDupFlag,
				splitPrefix: densifySplitPrefix, dupPrefix: densifyDupPrefix,
				keepFlag: densifyKeepFlag, keepPrefix: densifyKeepPrefix,
				blockTotals: densifyBlockTotals, compactScratch: densifyCompactScratch,
				randomSamples: densifyRandomSamples)
            let newCount = try await msplatDensify(denisifyParams, metalContext)

            numActive = newCount
            refreshViews()
            print("Densified: \(numPointsBefore) -> \(numActive) gaussians")
        }

        if step < stopSplitAt && step % resetInterval == refineEvery {
            //metalContext.gpuSync()
            let resetLogit: Float = -1.3862943611198906
            let op = opacities.floats()
            for i in 0..<opacities.numel where op[i] > resetLogit { op[i] = resetLogit }
            adamExpAvg[5].zero()
            adamExpAvgSq[5].zero()
            logErr("Opacity reset at step \(step)")
        }

        xysGradNorm.reset()
        visCounts.reset()
        max2DSize.reset()
    }

    // MARK: Export

    private var gaussianParams: GaussianParams {
        GaussianParams(means: means, scales: scales, quats: quats, featuresDc: featuresDc,
                       featuresRest: featuresRest, opacities: opacities,
                       scale: scale, translation: translation, keepCrs: keepCrs)
    }

    func save(_ filename: String, step: Int) throws {
        if (filename as NSString).pathExtension == "splat" { try saveSplat(filename) }
        else { try savePly(filename, step: step) }
        logErr("Saved \(filename)")
    }

    func savePly(_ filename: String, step: Int) throws {
        try saveGaussianPly(filename, gaussianParams, step: step)
    }

    func saveSplat(_ filename: String) throws {
        try saveGaussianSplat(filename, gaussianParams)
    }

    func loadPly(_ filename: String) throws -> Int {
        let g = try loadGaussianPly(filename, scale: scale, translation: translation, keepCrs: keepCrs, metalContext: metalContext)
        means = g.means
		scales = g.scales
		quats = g.quats
        featuresDc = g.featuresDc
		featuresRest = g.featuresRest
		opacities = g.opacities
        setupOptimizers()
        return g.step
    }

    // MARK: Checkpoints

    private static let ckptMagic: UInt32 = 0x4C50_534D  // "MSPL"
    private static let ckptVersion: UInt32 = 1

    private static func append<T: FixedWidthInteger>(_ d: inout Data, _ v: T) {
        withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
    }
    private static func append(_ d: inout Data, _ v: Float) { append(&d, v.bitPattern) }

    private static func writeTensor(_ d: inout Data, _ t: MTensor) {
        append(&d, UInt32(t.ndim))
        for i in 0..<t.ndim { append(&d, Int64(t.size(i))) }
        append(&d, UInt64(t.nbytes))
        d.append(contentsOf: UnsafeRawBufferPointer(start: t.rawPointer, count: t.nbytes))
    }

    private func readTensor(_ r: inout BinaryReader) throws -> MTensor {
        let ndim = Int(try r.read(UInt32.self))
        var shape: [Int] = []
        for _ in 0..<ndim { shape.append(Int(try r.read(Int64.self))) }
        let bytes = Int(try r.read(UInt64.self))
        let t = try metalContext.makeEmpty(shape, .float32)
        guard r.remaining >= bytes, bytes <= t.nbytes else { throw MsplatError("Checkpoint truncated") }
        r.data.withUnsafeBytes { raw in
            memcpy(t.rawPointer, raw.baseAddress!.advanced(by: r.offset), bytes)
        }
        try r.skip(bytes)
        return t
    }

    func saveCheckpoint(_ filename: String, step: Int) throws {
        //metalContext.gpuSync()

        var d = Data()
        Model.append(&d, Model.ckptMagic)
        Model.append(&d, Model.ckptVersion)
        Model.append(&d, UInt32(step))
        Model.append(&d, UInt32(numActive))
        Model.append(&d, UInt32(shDegree))
        Model.append(&d, UInt32(adamStepCount))
        for lr in adamLr { Model.append(&d, lr) }
        Model.append(&d, meansLrInit)
        Model.append(&d, meansLrFinal)

        for t in paramViews { Model.writeTensor(&d, t) }
        for g in 0..<Model.nAdamGroups { Model.writeTensor(&d, adamExpAvg[g]) }
        for g in 0..<Model.nAdamGroups { Model.writeTensor(&d, adamExpAvgSq[g]) }

        do { try d.write(to: URL(fileURLWithPath: filename)) }
        catch { throw MsplatError("Cannot open checkpoint file for writing: \(filename)") }
        print("Checkpoint saved: \(filename) (step \(step), \(numActive) gaussians, \(d.count / (1024 * 1024)) MB)")
    }

    func loadCheckpoint(_ filename: String) throws -> Int {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filename)) else {
            throw MsplatError("Cannot open checkpoint file: \(filename)")
        }
        var r = BinaryReader(data)
        let magic = try r.read(UInt32.self)
        let version = try r.read(UInt32.self)
        if magic != Model.ckptMagic { throw MsplatError("Not a valid msplat checkpoint file") }
        if version != Model.ckptVersion { throw MsplatError("Unsupported checkpoint version: \(version)") }

        let step = Int(try r.read(UInt32.self))
        let numPts = Int(try r.read(UInt32.self))
        _ = try r.read(UInt32.self) // shDegree (informational)
        let adamSteps = Int(try r.read(UInt32.self))
        for g in 0..<Model.nAdamGroups { adamLr[g] = try r.readF32() }
        meansLrInit = try r.readF32()
        meansLrFinal = try r.readF32()
        adamStepCount = adamSteps

        means = try readTensor(&r)
        scales = try readTensor(&r)
        quats = try readTensor(&r)
        featuresDc = try readTensor(&r)
        featuresRest = try readTensor(&r)
        opacities = try readTensor(&r)
        for g in 0..<Model.nAdamGroups { adamExpAvg[g] = try readTensor(&r) }
        for g in 0..<Model.nAdamGroups { adamExpAvgSq[g] = try readTensor(&r) }

        // Rebuild backing buffers (not setupOptimizers — that would zero the loaded state)
        numActive = numPts
        bufCapacity = numActive * 4
        meansBuf = try allocBuf(means)
        scalesBuf = try allocBuf(scales)
        quatsBuf = try allocBuf(quats)
        featuresDcBuf = try allocBuf(featuresDc)
        featuresRestBuf = try allocBuf(featuresRest)
        opacitiesBuf = try allocBuf(opacities)
        for g in 0..<Model.nAdamGroups {
            adamExpAvgBuf[g] = try allocBuf(adamExpAvg[g])
            adamExpAvgSqBuf[g] = try allocBuf(adamExpAvgSq[g])
        }
        try allocDensifyScratch(capacity: bufCapacity, frStride: featuresRest.numel / featuresRest.size(0))
        refreshViews()

        print("Checkpoint loaded: \(filename) (step \(step), \(numActive) gaussians)")
        return step
    }


	func prepareCam(_ cam: RuntimeCamera, step: Int) throws -> CameraRenderParams
	{
		let sf = Float(1.0)//Float(getDownscaleFactor(step))
		
		// Scale intrinsics by downscale factor
		let intrinsics = cam.meta.intrinsics.scaled(by: sf)

		let b = featuresRest.size(-2) + 1
		//	degrees depending on buffer size
		let degrees = b <= 1 ? 0 : b <= 4 ? 1 : b <= 9 ? 2 : b <= 16 ? 3 : 4
		let degreesToUse = min(step / shDegreeInterval, shDegree)
		
		//	need to remake cache
		let cameraRenderParams = try cam.meta.GetCameraRenderParams(metalContext: metalContext, cameraIntrinsics: intrinsics, shDegrees:degrees, shDegreesToUse:degreesToUse)
		return cameraRenderParams
	}




	func render(_ cam: RuntimeCamera, step: Int, channels:Int) async throws -> MTensor 
	{
		let cameraRenderParams = try prepareCam(cam, step: step)
		let renderParams = GetRenderParams(cameraRenderParams: cameraRenderParams, channels: channels)
		return try await msplatRender(renderParams, metalContext)
	}

	func GetSplatBuffers() -> SplatBuffers
	{
		.init(means3d: means, 
			  scales: scales, 
			  quats: quats, 
			  featuresDc: featuresDc, 
			  featuresRest: featuresRest, 
			  opacities: opacities)
	}
	
	func GetRenderParams(cameraRenderParams:CameraRenderParams,channels:Int) -> RenderParams
	{
		let splatBuffers = GetSplatBuffers()
		
		let renderParams = RenderParams(
			numPoints: means.size(0),
			splatBuffers:splatBuffers, 
			globScale: 1.0,
			viewmat: cameraRenderParams.viewMat.toFloat4x4(), 
			projmat: cameraRenderParams.projViewMat.toFloat4x4(),
			camPos: cameraRenderParams.camPos,
			intrinsics: cameraRenderParams.intrinsics,
			tileBounds: cameraRenderParams.tileBounds,
			clipThresh: 0.01,
			degree: UInt32(cameraRenderParams.degrees), 
			degreesToUse: UInt32(cameraRenderParams.degreesToUse),
			background: backgroundColor, 
			channels: UInt32(channels)
		)
		return renderParams
	}
	
	func GetSplats() throws -> [PlySplatPoint]
	{
		//	todo: use gpu buffers
		refreshViews()

		if featuresDc.shape[1] != 3
		{
			throw MsplatError("Expecting 3 sh0")
		}
		
		// Read from GPU tensors
		let positions = self.means.floats()
		let scalesData = self.scales.floats()
		let quatsData = self.quats.floats()
		let opacitysData = self.opacities.floats()
		let featureDcsData = self.featuresDc.floats()
		
		// Validation: Check for NaN/Inf values
		func validateSplatData() throws {
			var nanCount = 0
			var nanSources: [String: Int] = [:]
			
			for i in 0..<min(100, numPoints)
			{
				// Check first 100 for speed
				let pos = positions[float3: i]
				let scale = scalesData[float3: i]
				let quat = quatsData[float4: i]
				let opacity = opacitysData[i]
				let featureDc = featureDcsData[float3: i]
				
				if pos.x.isNaN || pos.y.isNaN || pos.z.isNaN {
					nanSources["position", default: 0] += 1
					nanCount += 1
				}
				if scale.x.isNaN || scale.y.isNaN || scale.z.isNaN {
					nanSources["scale", default: 0] += 1
					nanCount += 1
				}
				if quat.x.isNaN || quat.y.isNaN || quat.z.isNaN || quat.w.isNaN {
					nanSources["quaternion", default: 0] += 1
					nanCount += 1
				}
				if opacity.isNaN {
					nanSources["opacity", default: 0] += 1
					nanCount += 1
				}
				if featureDc.x.isNaN || featureDc.y.isNaN || featureDc.z.isNaN {
					nanSources["featureDc", default: 0] += 1
					nanCount += 1
				}
			}
			
			if nanCount > 0 {
				let report = nanSources.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
				throw MsplatError("GetSplats: Found \(nanCount) NaN values in first 100 splats: \(report)")
			}
		}
		
		//try validateSplatData()
		
		let points = (0..<numPoints).map
		{
			i in
			let pos = positions[float3: i]
			let scale = scalesData[float3: i]
			let quat = quatsData[float4: i]
			let opacity = opacitysData[i]
			let featureDc = featureDcsData[float3: i]
			let splat = PlySplatPoint(position: pos, quaternionWxyz: quat, scaleLog: scale, opacityLogit: opacity, shDc: featureDc)
			return splat
		}

		return points
	}
	
	func fullIteration(_ cam: RuntimeCamera, step: Int, gt: MTensor, ssimWeight: Float) async throws 
	{
        let cameraRenderParams = try prepareCam(cam, step: step)
        lastHeight = cameraRenderParams.intrinsics.height
		lastWidth = cameraRenderParams.intrinsics.width
        let numPoints = means.size(0)

        if !window2d.defined {
            window2d = try! metalContext.makeEmpty([11, 11])
            window2d.copyIn(createSSIMWindow(windowSize: 11, sigma: 1.5))
        }

		adamStepCount += 1
		// C++ std::pow(float, int) promotes to double; mirror that for identical rounding.
		let bc1 = Float(1.0 - pow(Double(adamBeta1), Double(adamStepCount)))
		let bc2 = Float(1.0 - pow(Double(adamBeta2), Double(adamStepCount)))
		let stepSizes = adamLr.map { $0 / bc1 }
		let bc2Sqrts = [Float](repeating: bc2.squareRoot(), count: Model.nAdamGroups)

		if !xysGradNorm.defined
		{
			xysGradNorm = try! metalContext.makeZeros([numPoints])
			visCounts = try! metalContext.makeZeros([numPoints])
			max2DSize = try! metalContext.makeZeros([numPoints])
		}

		let invMaxDim = 1.0 / Float(max(lastHeight, lastWidth))
		let lossInvN = 1.0 / Float(cameraRenderParams.intrinsics.height * cameraRenderParams.intrinsics.width * 3)

		let params = TrainParams(
			gt: gt,
			window2d: window2d, 
			ssimWeight: ssimWeight,
			lossInvN: lossInvN, 
			featuresRestBases: featuresRest.size(-2),
			adamParams: paramViews, 
			adamExpAvg: adamExpAvg, 
			adamExpAvgSq: adamExpAvgSq,
			adamStepSizes: stepSizes,
			adamBc2Sqrts: bc2Sqrts,
			adamBeta1: adamBeta1, 
			adamBeta2: adamBeta2, 
			adamEps: adamEps,
			visCounts: visCounts, 
			xysGradNorm: xysGradNorm, 
			max2dSize: max2DSize,
			invMaxDim: invMaxDim)

		let renderChannels = 3
		let renderParams = GetRenderParams(cameraRenderParams: cameraRenderParams, channels: renderChannels)
		/*
		let renderParams = RenderParams(
			numPoints: means.size(0), 
			means3d: means, 
			scales: scales, 
			globScale: 1.0,
			quats: quats,
			viewmat: viewMatrix,
			projmat: projViewMatrix,  // Combined projection×view matrix!
			
			//viewmat: cache.viewMat.toFloat4x4(), 
			//projmat: cache.projViewMat.toFloat4x4(),
			intrinsics: camSetup.intrinsics,
			tileBounds: camSetup.tileBounds,
			clipThresh: 0.01,
			degree: UInt32(camSetup.degree), 
			degreesToUse: UInt32(camSetup.degreesToUse),
			camPos: cameraRenderParams.camPos,
			featuresDc: featuresDc,
			featuresRest: featuresRest,
			opacities: opacities,
			background: backgroundColor, 
			channels: UInt32(channels)
		)
		let renderParams = RenderParams(cam, camSetup: s, channels: 3), params, metalContext)
		*/
		let (r, _) = try await msplatTrainStep(renderParams, params, metalContext)
        radii = r
    }
    
    
    /// Get detailed GPU memory usage breakdown
    func getGPUMemoryBreakdown() -> GPUMemoryBreakdown {
        var breakdown = GPUMemoryBreakdown()
        
        // Gaussian parameters (active views)
        breakdown.gaussianParams += means.nbytes
        breakdown.gaussianParams += scales.nbytes
        breakdown.gaussianParams += quats.nbytes
        breakdown.gaussianParams += featuresDc.nbytes
        breakdown.gaussianParams += featuresRest.nbytes
        breakdown.gaussianParams += opacities.nbytes
        
        // Gaussian backing buffers
        breakdown.gaussianBuffers += meansBuf.nbytes
        breakdown.gaussianBuffers += scalesBuf.nbytes
        breakdown.gaussianBuffers += quatsBuf.nbytes
        breakdown.gaussianBuffers += featuresDcBuf.nbytes
        breakdown.gaussianBuffers += featuresRestBuf.nbytes
        breakdown.gaussianBuffers += opacitiesBuf.nbytes
        
        // Adam state (active views)
        for g in 0..<Model.nAdamGroups {
            breakdown.adamState += adamExpAvg[g].nbytes
            breakdown.adamState += adamExpAvgSq[g].nbytes
        }
        
        // Adam backing buffers
        for g in 0..<Model.nAdamGroups {
            breakdown.adamBuffers += adamExpAvgBuf[g].nbytes
            breakdown.adamBuffers += adamExpAvgSqBuf[g].nbytes
        }
        
        // Densification scratch
        breakdown.densificationScratch += densifySplitFlag.nbytes
        breakdown.densificationScratch += densifyDupFlag.nbytes
        breakdown.densificationScratch += densifySplitPrefix.nbytes
        breakdown.densificationScratch += densifyDupPrefix.nbytes
        breakdown.densificationScratch += densifyKeepFlag.nbytes
        breakdown.densificationScratch += densifyKeepPrefix.nbytes
        breakdown.densificationScratch += densifyBlockTotals.nbytes
        breakdown.densificationScratch += densifyCompactScratch.nbytes
        breakdown.densificationScratch += densifyRandomSamples.nbytes
        
        // Training state
        breakdown.trainingState += xysGradNorm.nbytes
        breakdown.trainingState += visCounts.nbytes
        breakdown.trainingState += max2DSize.nbytes
        breakdown.trainingState += radii.nbytes
        
        // Other
        breakdown.other += window2d.nbytes
        
        return breakdown
    }
    
    /// Get total GPU memory usage in bytes
    var gpuMemoryUsage: Int {
        getGPUMemoryBreakdown().total
    }
    
    /// Get GPU memory usage in megabytes
    var gpuMemoryUsageMB: Double {
        Double(gpuMemoryUsage) / (1024.0 * 1024.0)
    }
    
    /// Get per-Gaussian memory usage estimate in bytes
    var bytesPerGaussian: Int {
        guard numActive > 0 else { return 0 }
        return gpuMemoryUsage / numActive
    }
    
    /// Get memory efficiency (percentage of allocated buffer capacity that's actually used)
    var memoryEfficiency: Double {
        guard bufCapacity > 0 else { return 0.0 }
        return Double(numActive) / Double(bufCapacity) * 100.0
    }
    
    /// Print detailed GPU memory report to console
    func printGPUMemoryReport() {
        let breakdown = getGPUMemoryBreakdown()
        print(breakdown.description)
        print("Active Gaussians:     \(numActive)")
        print("Buffer Capacity:      \(bufCapacity)")
        print("Memory Efficiency:    \(String(format: "%.1f%%", memoryEfficiency))")
        print("Bytes per Gaussian:   \(bytesPerGaussian)")
    }
}

