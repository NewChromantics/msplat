import Foundation
import NerfStudioData	//	plyHeader

private let C0 = 0.28209479177387814

struct GaussianParams {
    var means: MTensor, scales: MTensor, quats: MTensor
    var featuresDc: MTensor, featuresRest: MTensor, opacities: MTensor
    var scale: Float                          // CRS scale factor
    var translation: (Float, Float, Float)    // CRS translation
    var keepCrs: Bool
}

private func writeData(_ data: Data, to path: String) throws {
    do { try data.write(to: URL(fileURLWithPath: path)) }
    catch { throw MsplatError("Cannot write \(path): \(error.localizedDescription)") }
}

func saveGaussianPly(_ path: String, _ p: GaussianParams, step: Int) throws 
{
	throw MsplatError("save-ply should be grabbing a state")
	//  msplatGpuSync()

    let N = p.means.size(0)
    let numDc = p.featuresDc.size(1)
    let frBases = p.featuresRest.size(-2)
    let numFr = frBases * 3

    var header = "ply\nformat binary_little_endian 1.0\n"
    header += "comment msplat v\(step)\n"
    header += "element vertex \(N)\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    for i in 0..<numDc { header += "property float f_dc_\(i)\n" }
    for i in 0..<numFr { header += "property float f_rest_\(i)\n" }
    header += "property float opacity\n"
    header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    header += "end_header\n"

    let floatsPerRow = 3 + 3 + numDc + numFr + 1 + 3 + 4
    var body = [Float](repeating: 0, count: N * floatsPerRow)
    let mp = p.means.floats()
	let sp = p.scales.floats()
	let qp = p.quats.floats()
    let dp = p.featuresDc.floats()
	let op = p.opacities.floats() 
	let frp = p.featuresRest.floats()
    let tr = [p.translation.0, p.translation.1, p.translation.2]

    body.withUnsafeMutableBufferPointer 
	{
		out in
        for i in 0..<N 
		{
            var c = i * floatsPerRow
            for j in 0..<3 
			{
                out[c] = p.keepCrs ? (mp[i * 3 + j] / p.scale + tr[j]) : mp[i * 3 + j]
				c += 1
            }
            c += 3 // normals = 0
            for j in 0..<numDc 
			{
				out[c] = dp[i * numDc + j]; 
				c += 1 
			}
            // Transpose [frBases, 3] → [3, frBases] for PLY convention
            for ch in 0..<3 
			{
                for b in 0..<frBases { out[c] = frp[i * frBases * 3 + b * 3 + ch]; c += 1 }
            }
            out[c] = op[i]
			c += 1
            for j in 0..<3 
			{
                out[c] = p.keepCrs ? logf(expf(sp[i * 3 + j]) / p.scale) : sp[i * 3 + j]; c += 1
            }
            for j in 0..<4 
			{
				out[c] = qp[i * 4 + j]
				c += 1 
			}
        }
    }

    var data = Data(header.utf8)
    body.withUnsafeBytes { data.append(contentsOf: $0) }  // little-endian host
    try writeData(data, to: path)
}


@inline(__always)
func clampToUInt8(_ v: Float) -> UInt8 {
	if v.isNaN { return 0 }
	return UInt8(min(max(v, 0), 255))
}

func saveGaussianSplat(_ path: String, _ p: GaussianParams) throws {
 
    let N = p.means.size(0)
    let mp = p.means.floats(), sp = p.scales.floats(), qp = p.quats.floats()
    let dp = p.featuresDc.floats(), op = p.opacities.floats()
    let tr = [p.translation.0, p.translation.1, p.translation.2]

    // Sort by size × opacity, largest first
    var order = [Float](repeating: 0, count: N)
    for i in 0..<N {
        var s = expf(sp[i * 3]) + expf(sp[i * 3 + 1]) + expf(sp[i * 3 + 2])
        if p.keepCrs { s /= p.scale }
        order[i] = s / (1.0 + expf(-op[i]))
    }
    let idx = (0..<N).sorted { order[$0] > order[$1] }

    let rowBytes = 32 // 3f pos + 3f scale + 4B rgba + 4B quat
    var out = Data(count: N * rowBytes)
    out.withUnsafeMutableBytes { raw in
        let base = raw.baseAddress!
        for (row, i) in idx.enumerated() {
            let r = base.advanced(by: row * rowBytes)
            for j in 0..<3 {
                let m = p.keepCrs ? (mp[i * 3 + j] / p.scale + tr[j]) : mp[i * 3 + j]
                r.storeBytes(of: m.bitPattern.littleEndian, toByteOffset: j * 4, as: UInt32.self)
            }
            for j in 0..<3 {
                let s = p.keepCrs ? (expf(sp[i * 3 + j]) / p.scale) : expf(sp[i * 3 + j])
                r.storeBytes(of: s.bitPattern.littleEndian, toByteOffset: 12 + j * 4, as: UInt32.self)
            }
            for j in 0..<3 {
                let v = (Double(dp[i * 3 + j]) * C0 + 0.5) * 255.0
                let b: UInt8 = v.isNaN ? 0 : UInt8(min(max(v, 0.0), 255.0))
                r.storeBytes(of: b, toByteOffset: 24 + j, as: UInt8.self)
            }
            let sig = 1.0 / (1.0 + expf(-op[i]))
            r.storeBytes(of: clampToUInt8(sig * 255.0), toByteOffset: 27, as: UInt8.self)
            for j in 0..<4 {
                r.storeBytes(of: clampToUInt8(qp[i * 4 + j] * 128.0 + 128.0), toByteOffset: 28 + j, as: UInt8.self)
            }
        }
    }
    try writeData(out, to: path)
}

struct LoadedGaussians {
    var means: MTensor, scales: MTensor, quats: MTensor
    var featuresDc: MTensor, featuresRest: MTensor, opacities: MTensor
    var step: Int
}

func loadGaussianPly(_ path: String, scale: Float, translation: (Float, Float, Float),
					 keepCrs: Bool,metalContext:MetalContext) throws -> LoadedGaussians {

    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        throw MsplatError("Cannot open PLY file: \(path)")
    }
    guard let header = plyHeader(data), let first = header.lines.first, first.contains("ply") else {
        throw MsplatError("Not a PLY file: \(path)")
    }

    var numPoints = 0, step = 0, numDc = 0, numFr = 0
    // C++ only recognised the first prefix, which saveGaussianPly never writes;
    // "comment msplat v<step>" (what it does write) is accepted too.
    let iterPrefixes = ["comment Generated by msplat at iteration ", "comment msplat v"]
    for line in header.lines.dropFirst(2) where line != "end_header" {
        for pre in iterPrefixes where line.hasPrefix(pre) {
            step = Int(line.dropFirst(pre.count).trimmingCharacters(in: .whitespaces)) ?? step
        }
        if line.hasPrefix("element vertex ") {
            numPoints = Int(line.dropFirst("element vertex ".count).trimmingCharacters(in: .whitespaces)) ?? 0
        }
        if line.hasPrefix("property float f_dc_") { numDc += 1 }
        if line.hasPrefix("property float f_rest_") { numFr += 1 }
    }
    if numPoints == 0 { throw MsplatError("PLY has no vertices") }
    let frBases = numFr / 3

    // Row: xyz(3) normals(3) f_dc(numDc) f_rest(numFr) opacity(1) scale(3) rot(4)
    let rowFloats = 3 + 3 + numDc + numFr + 1 + 3 + 4
    guard data.count - header.bodyOffset >= numPoints * rowFloats * 4 else {
        throw MsplatError("PLY truncated: \(path)")
    }

    var meansRaw = [Float](repeating: 0, count: numPoints * 3)
    var dcRaw = [Float](repeating: 0, count: numPoints * numDc)
    var frRaw = [Float](repeating: 0, count: numPoints * numFr)
    var opRaw = [Float](repeating: 0, count: numPoints)
    var scRaw = [Float](repeating: 0, count: numPoints * 3)
    var qtRaw = [Float](repeating: 0, count: numPoints * 4)

    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!.advanced(by: header.bodyOffset)
        func f(_ i: Int) -> Float {
            Float(bitPattern: UInt32(littleEndian: base.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)))
        }
        for i in 0..<numPoints {
            var c = i * rowFloats
            for j in 0..<3 { meansRaw[i * 3 + j] = f(c); c += 1 }
            c += 3 // normals
            for j in 0..<numDc { dcRaw[i * numDc + j] = f(c); c += 1 }
            for j in 0..<numFr { frRaw[i * numFr + j] = f(c); c += 1 }
            opRaw[i] = f(c); c += 1
            for j in 0..<3 { scRaw[i * 3 + j] = f(c); c += 1 }
            for j in 0..<4 { qtRaw[i * 4 + j] = f(c); c += 1 }
        }
    }

    if keepCrs {
        let tr = [translation.0, translation.1, translation.2]
        for i in 0..<numPoints {
            for j in 0..<3 { meansRaw[i * 3 + j] = (meansRaw[i * 3 + j] - tr[j]) * scale }
        }
        for i in 0..<(numPoints * 3) { scRaw[i] = logf(scale * expf(scRaw[i])) }
    }

    func upload(_ shape: [Int], _ src: [Float]) throws -> MTensor {
		let t = try metalContext.makeEmpty(shape, .float32)
        t.copyIn(src)
        return t
    }

    // Transpose featuresRest: PLY [N, 3, frBases] → internal [N, frBases, 3]
	let fr = try metalContext.makeEmpty([numPoints, frBases, 3], .float32)
    let frOut = fr.floats()
    for i in 0..<numPoints {
        for ch in 0..<3 {
            for b in 0..<frBases {
                frOut[i * frBases * 3 + b * 3 + ch] = frRaw[i * numFr + ch * frBases + b]
            }
        }
    }

    return LoadedGaussians(
        means: try upload([numPoints, 3], meansRaw),
        scales: try upload([numPoints, 3], scRaw),
        quats: try upload([numPoints, 4], qtRaw),
        featuresDc: try upload([numPoints, numDc], dcRaw),
        featuresRest: fr,
        opacities: try upload([numPoints, 1], opRaw),
        step: step)
}
