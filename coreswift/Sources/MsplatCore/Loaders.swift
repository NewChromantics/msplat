// Port of loaders/load_colmap.cpp, load_nerfstudio.cpp, load_polycam.cpp and
// load_ply.cpp (point-cloud reader). nlohmann/json is replaced by JSONSerialization.

import Foundation

// MARK: - Helpers

/// Little-endian binary reader over Data.
struct BinaryReader {
    let data: Data
    var offset = 0

    init(_ data: Data) { self.data = data }

    var remaining: Int { data.count - offset }

    mutating func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
        let n = MemoryLayout<T>.size
        guard remaining >= n else { throw MsplatError("Unexpected end of file") }
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
        guard n >= 0, remaining >= n else { throw MsplatError("Unexpected end of file") }
        offset += n
    }
}

/// Byte-wise string ordering (std::string operator<).
@inline(__always)
func byteLess(_ a: String, _ b: String) -> Bool {
    a.utf8.lexicographicallyPrecedes(b.utf8)
}

func fileExists(_ path: String) -> Bool { FileManager.default.fileExists(atPath: path) }

func joinPath(_ a: String, _ b: String) -> String {
    URL(fileURLWithPath: a).appendingPathComponent(b).path
}

/// float → uint8 with clamping (a raw C cast is UB out of range; Swift would trap).
@inline(__always)
func clampToUInt8(_ v: Float) -> UInt8 {
    if v.isNaN { return 0 }
    return UInt8(min(max(v, 0), 255))
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

// MARK: - COLMAP

private struct ColmapCamera {
    var id: UInt32 = 0
    var model = 0
    var width = 0, height = 0
    var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
    var k1: Float = 0, k2: Float = 0, p1: Float = 0, p2: Float = 0
}

private struct ColmapImage {
    var camId: UInt32 = 0
    var quat: [Double] = []  // w, x, y, z
    var t: [Double] = []     // world-to-camera translation
    var filename = ""
}

private func readCamerasBin(_ path: String) throws -> [UInt32: ColmapCamera] {
    var f = BinaryReader(try Data(contentsOf: URL(fileURLWithPath: path)))
    let n = try f.read(UInt64.self)
    var cams: [UInt32: ColmapCamera] = [:]
    for _ in 0..<n {
        var c = ColmapCamera()
        c.id = try f.read(UInt32.self)
        c.model = Int(try f.read(UInt32.self))
        c.width = Int(try f.read(UInt64.self))
        c.height = Int(try f.read(UInt64.self))
        func rd() throws -> Float { Float(try f.readF64()) }
        switch c.model {
        case 0: // SIMPLE_PINHOLE
            c.fx = try rd(); c.fy = c.fx; c.cx = try rd(); c.cy = try rd()
        case 1: // PINHOLE
            c.fx = try rd(); c.fy = try rd(); c.cx = try rd(); c.cy = try rd()
        case 2: // SIMPLE_RADIAL
            c.fx = try rd(); c.fy = c.fx; c.cx = try rd(); c.cy = try rd(); c.k1 = try rd()
        case 3: // RADIAL
            c.fx = try rd(); c.fy = c.fx; c.cx = try rd(); c.cy = try rd(); c.k1 = try rd(); c.k2 = try rd()
        case 4: // OPENCV
            c.fx = try rd(); c.fy = try rd(); c.cx = try rd(); c.cy = try rd()
            c.k1 = try rd(); c.k2 = try rd(); c.p1 = try rd(); c.p2 = try rd()
        default:
            throw MsplatError("Unsupported COLMAP camera model: \(c.model)")
        }
        cams[c.id] = c
    }
    return cams
}

private func readImagesBin(_ path: String) throws -> [ColmapImage] {
    var f = BinaryReader(try Data(contentsOf: URL(fileURLWithPath: path)))
    let n = try f.read(UInt64.self)
    var images: [ColmapImage] = []
    images.reserveCapacity(Int(n))
    for _ in 0..<n {
        var img = ColmapImage()
        _ = try f.read(UInt32.self) // image id
        img.quat = [try f.readF64(), try f.readF64(), try f.readF64(), try f.readF64()]
        img.t = [try f.readF64(), try f.readF64(), try f.readF64()]
        img.camId = try f.read(UInt32.self)
        img.filename = try f.readCString()
        let numPts2D = try f.read(UInt64.self)
        try f.skip(Int(numPts2D) * 24)
        images.append(img)
    }
    return images
}

/// Quaternion [w,x,y,z] → row-major 3x3 rotation.
private func quatToRotMat(_ q: [Double]) -> [Float] {
    var w = q[0], x = q[1], y = q[2], z = q[3]
    let n = (w * w + x * x + y * y + z * z).squareRoot()
    w /= n; x /= n; y /= n; z /= n
    return [
        Float(1 - 2 * (y * y + z * z)), Float(2 * (x * y - w * z)), Float(2 * (x * z + w * y)),
        Float(2 * (x * y + w * z)), Float(1 - 2 * (x * x + z * z)), Float(2 * (y * z - w * x)),
        Float(2 * (x * z - w * y)), Float(2 * (y * z + w * x)), Float(1 - 2 * (x * x + y * y)),
    ]
}

/// w2c rotation + translation → 4x4 c2w row-major with OpenGL Y/Z flip.
private func w2cToCamToWorld(_ quat: [Double], _ t: [Double]) -> [Float] {
    let R = quatToRotMat(quat)
    let Ri: [Float] = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]
    let t0 = Float(t[0]), t1 = Float(t[1]), t2 = Float(t[2])
    let Ti: [Float] = [
        -(Ri[0] * t0 + Ri[1] * t1 + Ri[2] * t2),
        -(Ri[3] * t0 + Ri[4] * t1 + Ri[5] * t2),
        -(Ri[6] * t0 + Ri[7] * t1 + Ri[8] * t2),
    ]
    return [
        Ri[0], -Ri[1], -Ri[2], Ti[0],
        Ri[3], -Ri[4], -Ri[5], Ti[1],
        Ri[6], -Ri[7], -Ri[8], Ti[2],
        0, 0, 0, 1,
    ]
}

func loadColmap(_ projectRoot: String, imageSourcePath: String = "") throws -> InputData {
    let sparseDir = fileExists(joinPath(projectRoot, "cameras.bin"))
        ? projectRoot : joinPath(joinPath(projectRoot, "sparse"), "0")
    let imageDir = !imageSourcePath.isEmpty ? imageSourcePath
        : fileExists(joinPath(projectRoot, "images")) ? joinPath(projectRoot, "images")
        : projectRoot

    let cameras = try readCamerasBin(sparseDir + "/cameras.bin")
    var images = try readImagesBin(sparseDir + "/images.bin")
    images.sort { byteLess($0.filename, $1.filename) }

    let data = InputData()
    data.cameras.reserveCapacity(images.count)
    for img in images {
        guard let cc = cameras[img.camId] else { continue }
        let cam = Camera()
        cam.width = cc.width; cam.height = cc.height
        cam.fx = cc.fx; cam.fy = cc.fy; cam.cx = cc.cx; cam.cy = cc.cy
        cam.k1 = cc.k1; cam.k2 = cc.k2; cam.p1 = cc.p1; cam.p2 = cc.p2
        cam.filePath = imageDir + "/" + img.filename
        cam.camToWorld = w2cToCamToWorld(img.quat, img.t)
        data.cameras.append(cam)
    }

    let ptsPath = sparseDir + "/points3D.bin"
    if fileExists(ptsPath) {
        data.points = try readColmapPoints(ptsPath)
    } else if fileExists(sparseDir + "/points3D.ply") {
        data.points = try readPly(sparseDir + "/points3D.ply")
    }

    autoScaleAndCenter(data)
    return data
}

// MARK: - Nerfstudio

/// Try common image extensions if the file doesn't exist.
private func resolveImagePath(_ path: String) -> String {
    if fileExists(path) { return path }
    for ext in [".png", ".jpg", ".jpeg", ".JPG"] where fileExists(path + ext) { return path + ext }
    return path
}

func loadNerfstudio(_ projectRoot: String) throws -> InputData {
    guard let j = try loadJSON(joinPath(projectRoot, "transforms.json")) as? [String: Any] else {
        throw MsplatError("transforms.json: expected a JSON object")
    }

    // Global defaults (overridden per-frame if present)
    let gW = num(j, "w", 0), gH = num(j, "h", 0)
    let gFx = num(j, "fl_x", Float(0)), gFy = num(j, "fl_y", Float(0))
    let gCx = num(j, "cx", Float(0)), gCy = num(j, "cy", Float(0))
    let gK1 = num(j, "k1", Float(0)), gK2 = num(j, "k2", Float(0)), gK3 = num(j, "k3", Float(0))
    let gP1 = num(j, "p1", Float(0)), gP2 = num(j, "p2", Float(0))

    let data = InputData()
    for frame in (j["frames"] as? [[String: Any]]) ?? [] {
        let cam = Camera()
        cam.width = num(frame, "w", gW); cam.height = num(frame, "h", gH)
        cam.fx = num(frame, "fl_x", gFx); cam.fy = num(frame, "fl_y", gFy)
        cam.cx = num(frame, "cx", gCx); cam.cy = num(frame, "cy", gCy)
        cam.k1 = num(frame, "k1", gK1); cam.k2 = num(frame, "k2", gK2)
        cam.k3 = num(frame, "k3", gK3)
        cam.p1 = num(frame, "p1", gP1); cam.p2 = num(frame, "p2", gP2)

        guard let fp = frame["file_path"] as? String, !fp.isEmpty else {
            throw MsplatError("transforms.json: frame missing file_path")
        }
        if fp.hasPrefix("/") {
            cam.filePath = resolveImagePath(fp)
        } else if fp.hasPrefix(".") {
            // C++ resolves "./x" against the process CWD. Keep that, but fall back to
            // the project root so "./images/..." works from any working directory.
            let cwdPath = resolveImagePath(fp)
            cam.filePath = fileExists(cwdPath) ? cwdPath : resolveImagePath(joinPath(projectRoot, fp))
        } else {
            cam.filePath = resolveImagePath(joinPath(projectRoot, fp))
        }

        guard let m = readMatrix4(frame["transform_matrix"]) else {
            throw MsplatError("transforms.json: frame missing transform_matrix")
        }
        cam.camToWorld = m
        data.cameras.append(cam)
    }

    data.cameras.sort { byteLess($0.filePath, $1.filePath) }

    if var p = j["ply_file_path"] as? String {
        if !p.hasPrefix("/") { p = joinPath(projectRoot, p) }
        if fileExists(p) { data.points = try readPly(p) }
    }
    if data.points.count == 0 {
        for rel in ["sparse/0/points3D.ply", "points3D.ply"] {
            let path = joinPath(projectRoot, rel)
            if fileExists(path) { data.points = try readPly(path); break }
        }
    }

    autoScaleAndCenter(data)
    return data
}

// MARK: - Polycam

private func findImage(_ dir: String, _ stem: String) -> String {
    for ext in [".png", ".jpg", ".jpeg", ".JPG"] {
        let p = joinPath(dir, stem + ext)
        if fileExists(p) { return p }
    }
    return joinPath(dir, stem + ".png")
}

private func polycamIntrinsics(_ cam: Camera, _ j: [String: Any]) {
    cam.width = num(j, "width", 0)
    cam.height = num(j, "height", 0)
    cam.fx = num(j, "fx", Float(0)); cam.fy = num(j, "fy", Float(0))
    cam.cx = num(j, "cx", Float(cam.width) / 2.0)
    cam.cy = num(j, "cy", Float(cam.height) / 2.0)
}

func loadPolycam(_ projectRoot: String) throws -> InputData {
    let data = InputData()
    let keyframesDir = joinPath(joinPath(projectRoot, "keyframes"), "corrected_cameras")
    let imagesDir = joinPath(joinPath(projectRoot, "keyframes"), "corrected_images")

    if fileExists(keyframesDir) {
        // Layout 1: individual camera JSONs
        let files = try FileManager.default.contentsOfDirectory(atPath: keyframesDir)
            .filter { ($0 as NSString).pathExtension == "json" }
            .map { joinPath(keyframesDir, $0) }
            .sorted(by: byteLess)

        for jp in files {
            guard let j = try loadJSON(jp) as? [String: Any] else { continue }
            let cam = Camera()
            polycamIntrinsics(cam, j)

            var R = [Float](repeating: 0, count: 9)
            for r in 0..<3 { for c in 0..<3 { R[r * 3 + c] = num(j, "R_\(r)\(c)", Float(0)) } }
            let T: [Float] = [num(j, "t_20", Float(0)), num(j, "t_21", Float(0)), num(j, "t_22", Float(0))]

            // c2w with OpenGL Y/Z flip
            cam.camToWorld = [
                 R[0],  R[1],  R[2],  T[0],
                -R[3], -R[4], -R[5], -T[1],
                -R[6], -R[7], -R[8], -T[2],
                 0,     0,     0,     1,
            ]
            let stem = ((jp as NSString).lastPathComponent as NSString).deletingPathExtension
            cam.filePath = findImage(imagesDir, stem)
            data.cameras.append(cam)
        }
    } else {
        // Layout 2: single cameras.json
        let j = try loadJSON(joinPath(projectRoot, "cameras.json"))
        let frames: [[String: Any]]
        if let obj = j as? [String: Any], let f = obj["frames"] as? [[String: Any]] {
            frames = f
        } else if let arr = j as? [[String: Any]] {
            frames = arr
        } else {
            frames = []
        }
        for frame in frames {
            let cam = Camera()
            polycamIntrinsics(cam, frame)
            if let m = readMatrix4(frame["transform_matrix"]) { cam.camToWorld = m }
            cam.filePath = (frame["file_path"] as? String) ?? ""
            if !cam.filePath.isEmpty && !cam.filePath.hasPrefix("/") {
                cam.filePath = joinPath(projectRoot, cam.filePath)
            }
            data.cameras.append(cam)
        }
    }

    for rel in ["keyframes/point_cloud.ply", "point_cloud.ply", "sparse.ply"] {
        let path = joinPath(projectRoot, rel)
        if fileExists(path) { data.points = try readPly(path); break }
    }

    autoScaleAndCenter(data)
    return data
}

// MARK: - PLY point cloud

private enum PlyType {
    case float32, float64, uint8, uint16, int32, unknown

    init(_ t: String) {
        switch t {
        case "float", "float32": self = .float32
        case "double", "float64": self = .float64
        case "uchar", "uint8": self = .uint8
        case "ushort", "uint16": self = .uint16
        case "int", "int32": self = .int32
        default: self = .unknown
        }
    }

    var size: Int {
        switch self {
        case .float32, .int32: return 4
        case .float64: return 8
        case .uint8: return 1
        case .uint16: return 2
        case .unknown: return 0
        }
    }
}

private struct PlyProp {
    var name: String
    var type: PlyType
    var offset: Int
}

/// Splits a PLY file into header lines and the byte offset where the body starts.
func plyHeader(_ data: Data) -> (lines: [String], bodyOffset: Int)? {
    var lines: [String] = []
    var start = 0
    let bytes = [UInt8](data.prefix(64 * 1024 * 16))  // headers are small
    var i = 0
    while i < bytes.count {
        if bytes[i] == 0x0A {
            var line = String(decoding: bytes[start..<i], as: UTF8.self)
            if line.hasSuffix("\r") { line.removeLast() }
            lines.append(line)
            start = i + 1
            if line == "end_header" { return (lines, start) }
        }
        i += 1
    }
    return nil
}

func readPly(_ path: String) throws -> Points {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        throw MsplatError("Cannot open PLY file: \(path)")
    }
    guard let header = plyHeader(data), let first = header.lines.first, first.contains("ply") else {
        throw MsplatError("Not a PLY file: \(path)")
    }
    let lines = header.lines, bodyOffset = header.bodyOffset

    var binary = false, ascii = false
    var numVertices = 0
    var props: [PlyProp] = []
    var vertexBytes = 0
    var inVertexElement = false

    for line in lines.dropFirst() where line != "end_header" {
        let tok = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard let head = tok.first else { continue }
        switch head {
        case "format":
            let fmt = tok.count > 1 ? tok[1] : ""
            binary = fmt == "binary_little_endian"
            ascii = fmt == "ascii"
        case "element":
            inVertexElement = tok.count > 1 && tok[1] == "vertex"
            if inVertexElement, tok.count > 2 { numVertices = Int(tok[2]) ?? 0 }
        case "property":
            // Only vertex properties. (The C++ also collected properties of any element
            // declared after "vertex", which corrupts the vertex stride if one exists.)
            if inVertexElement && numVertices > 0 && tok.count > 2 {
                let t = PlyType(tok[1])
                props.append(PlyProp(name: tok[2], type: t, offset: vertexBytes))
                vertexBytes += t.size
            }
        default:
            break
        }
    }

    if numVertices == 0 { throw MsplatError("PLY has no vertices") }
    if !binary && !ascii { throw MsplatError("Unsupported PLY format (need ascii or binary_little_endian)") }

    func find(_ n: String) -> Int? { props.firstIndex { $0.name == n } }
    guard let ix = find("x"), let iy = find("y"), let iz = find("z") else {
        throw MsplatError("PLY missing x/y/z properties")
    }
    let ir = find("red"), ig = find("green"), ib = find("blue")
    let hasColor = ir != nil && ig != nil && ib != nil

    var pts = Points()
    pts.count = numVertices
    pts.xyz = [Float](repeating: 0, count: numVertices * 3)
    pts.rgb = [UInt8](repeating: 128, count: numVertices * 3) // default grey

    if binary {
        guard data.count - bodyOffset >= numVertices * vertexBytes else {
            throw MsplatError("PLY truncated: \(path)")
        }
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: bodyOffset)
            for i in 0..<numVertices {
                let rec = base.advanced(by: i * vertexBytes)
                func readFloat(_ pi: Int) -> Float {
                    let p = props[pi]
                    switch p.type {
                    case .float32: return Float(bitPattern: UInt32(littleEndian: rec.loadUnaligned(fromByteOffset: p.offset, as: UInt32.self)))
                    case .float64: return Float(Double(bitPattern: UInt64(littleEndian: rec.loadUnaligned(fromByteOffset: p.offset, as: UInt64.self))))
                    default: return 0
                    }
                }
                func readU8(_ pi: Int) -> UInt8 {
                    let p = props[pi]
                    switch p.type {
                    case .uint8: return rec.load(fromByteOffset: p.offset, as: UInt8.self)
                    case .uint16: return UInt8(UInt16(littleEndian: rec.loadUnaligned(fromByteOffset: p.offset, as: UInt16.self)) >> 8)
                    case .float32: return clampToUInt8(readFloat(pi) * 255.0)
                    default: return 128
                    }
                }
                pts.xyz[i * 3 + 0] = readFloat(ix)
                pts.xyz[i * 3 + 1] = readFloat(iy)
                pts.xyz[i * 3 + 2] = readFloat(iz)
                if hasColor {
                    pts.rgb[i * 3 + 0] = readU8(ir!)
                    pts.rgb[i * 3 + 1] = readU8(ig!)
                    pts.rgb[i * 3 + 2] = readU8(ib!)
                }
            }
        }
    } else {
        let body = String(decoding: data[(data.startIndex + bodyOffset)...], as: UTF8.self)
        var lineIter = body.split(separator: "\n", omittingEmptySubsequences: false).makeIterator()
        for i in 0..<numVertices {
            guard let line = lineIter.next() else { throw MsplatError("PLY truncated: \(path)") }
            let tokens = line.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\r" })
            func f(_ idx: Int) throws -> Float {
                guard idx < tokens.count, let v = Float(tokens[idx]) else {
                    throw MsplatError("PLY parse error at vertex \(i): \(path)")
                }
                return v
            }
            pts.xyz[i * 3 + 0] = try f(ix)
            pts.xyz[i * 3 + 1] = try f(iy)
            pts.xyz[i * 3 + 2] = try f(iz)
            if hasColor {
                let rv = try f(ir!), gv = try f(ig!), bv = try f(ib!)
                let isFloat = props[ir!].type == .float32 || props[ir!].type == .float64
                let s: Float = isFloat ? 255.0 : 1.0
                pts.rgb[i * 3 + 0] = clampToUInt8(rv * s)
                pts.rgb[i * 3 + 1] = clampToUInt8(gv * s)
                pts.rgb[i * 3 + 2] = clampToUInt8(bv * s)
            }
        }
    }
    return pts
}

/// COLMAP points3D.bin reader.
func readColmapPoints(_ path: String) throws -> Points {
    guard let d = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        throw MsplatError("Cannot open points3D.bin: \(path)")
    }
    var f = BinaryReader(d)
    let n = Int(try f.read(UInt64.self))
    var pts = Points()
    pts.count = n
    pts.xyz = [Float](repeating: 0, count: n * 3)
    pts.rgb = [UInt8](repeating: 0, count: n * 3)
    for i in 0..<n {
        _ = try f.read(UInt64.self) // point id
        pts.xyz[i * 3 + 0] = Float(try f.readF64())
        pts.xyz[i * 3 + 1] = Float(try f.readF64())
        pts.xyz[i * 3 + 2] = Float(try f.readF64())
        pts.rgb[i * 3 + 0] = try f.read(UInt8.self)
        pts.rgb[i * 3 + 1] = try f.read(UInt8.self)
        pts.rgb[i * 3 + 2] = try f.read(UInt8.self)
        _ = try f.readF64() // reprojection error
        let trackLen = Int(try f.read(UInt64.self))
        try f.skip(trackLen * 8) // (image_id u32, point2d_idx u32) per entry
    }
    return pts
}
