// Replacement for include/kdtree_tensor.hpp + nanoflann.
// Exact k-nearest-neighbour search, so per-point scales are identical to the C++
// result (distances are computed in float in the same dimension order as nanoflann's
// L2_Simple_Adaptor). Queries run in parallel.

import Foundation

struct PointsKDTree {
    private let pts: UnsafeBufferPointer<Float>  // count*3, borrowed
    private let count: Int
    private var idx: [Int32]
    // Implicit tree nodes: (lo, hi) ranges in `idx`, split dim/value, children.
    private var nodes: [Node] = []
    private let leafSize = 10

    private struct Node {
        var lo: Int32, hi: Int32
        var left: Int32 = -1, right: Int32 = -1
        var dim: UInt8 = 0
        var split: Float = 0
        var bbMin: SIMD3<Float>, bbMax: SIMD3<Float>
    }

    init(points: UnsafeBufferPointer<Float>, count: Int) {
        pts = points
        self.count = count
        idx = (0..<Int32(count)).map { $0 }
        nodes.reserveCapacity(max(1, 2 * count / leafSize + 1))
        if count > 0 { _ = build(0, count) }
    }

    @inline(__always) private func p(_ i: Int32) -> SIMD3<Float> {
        let b = Int(i) * 3
        return SIMD3(pts[b], pts[b + 1], pts[b + 2])
    }

    private mutating func build(_ lo: Int, _ hi: Int) -> Int32 {
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in lo..<hi { let q = p(idx[i]); mn = pointwiseMin(mn, q); mx = pointwiseMax(mx, q) }
        let nodeIndex = Int32(nodes.count)
        nodes.append(Node(lo: Int32(lo), hi: Int32(hi), bbMin: mn, bbMax: mx))
        if hi - lo <= leafSize { return nodeIndex }

        let ext = mx - mn
        let dim = ext.x >= ext.y ? (ext.x >= ext.z ? 0 : 2) : (ext.y >= ext.z ? 1 : 2)
        let mid = (lo + hi) / 2
        // nth_element on the split dimension
        let pts = self.pts
        idx.withUnsafeMutableBufferPointer { buf in
            PointsKDTree.quickselect(buf, pts, lo, hi - 1, mid, dim)
        }
        let split = pts[Int(idx[mid]) * 3 + dim]
        let l = build(lo, mid)
        let r = build(mid, hi)
        nodes[Int(nodeIndex)].left = l
        nodes[Int(nodeIndex)].right = r
        nodes[Int(nodeIndex)].dim = UInt8(dim)
        nodes[Int(nodeIndex)].split = split
        return nodeIndex
    }

    private static func quickselect(_ a: UnsafeMutableBufferPointer<Int32>, _ pts: UnsafeBufferPointer<Float>,
                                    _ lo0: Int, _ hi0: Int, _ k: Int, _ dim: Int) {
        var lo = lo0, hi = hi0
        while lo < hi {
            let pivot = pts[Int(a[(lo + hi) / 2]) * 3 + dim]
            var i = lo, j = hi
            while i <= j {
                while pts[Int(a[i]) * 3 + dim] < pivot { i += 1 }
                while pts[Int(a[j]) * 3 + dim] > pivot { j -= 1 }
                if i <= j { a.swapAt(i, j); i += 1; j -= 1 }
            }
            if k <= j { hi = j } else if k >= i { lo = i } else { return }
        }
    }

    @inline(__always) private static func dist2(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        // Same accumulation order as nanoflann L2_Simple_Adaptor.
        var r: Float = 0
        var d = a.x - b.x; r += d * d
        d = a.y - b.y; r += d * d
        d = a.z - b.z; r += d * d
        return r
    }

    @inline(__always) private static func boxDist2(_ q: SIMD3<Float>, _ mn: SIMD3<Float>, _ mx: SIMD3<Float>) -> Float {
        let d = pointwiseMax(pointwiseMax(mn - q, q - mx), SIMD3<Float>(repeating: 0))
        return (d * d).sum()
    }

    /// Squared distances of the k nearest points to `q`, ascending (includes the point itself).
    func knn(_ q: SIMD3<Float>, k: Int, into best: inout [Float], stack: inout [Int32]) {
        best.removeAll(keepingCapacity: true)
        var worst = Float.infinity
        stack.removeAll(keepingCapacity: true)
        stack.append(0)
        while let ni = stack.popLast() {
            let n = nodes[Int(ni)]
            if best.count == k && PointsKDTree.boxDist2(q, n.bbMin, n.bbMax) > worst { continue }
            if n.left < 0 {
                for i in Int(n.lo)..<Int(n.hi) {
                    let d = PointsKDTree.dist2(q, p(idx[i]))
                    if best.count < k {
                        // insertion sort, ascending
                        var j = best.count
                        best.append(d)
                        while j > 0 && best[j - 1] > d { best[j] = best[j - 1]; j -= 1 }
                        best[j] = d
                        if best.count == k { worst = best[k - 1] }
                    } else if d < worst {
                        var j = k - 1
                        while j > 0 && best[j - 1] > d { best[j] = best[j - 1]; j -= 1 }
                        best[j] = d
                        worst = best[k - 1]
                    }
                }
            } else {
                // Visit the nearer child first (push it last).
                let qd = n.dim == 0 ? q.x : (n.dim == 1 ? q.y : q.z)
                if qd < n.split { stack.append(n.right); stack.append(n.left) }
                else { stack.append(n.left); stack.append(n.right) }
            }
        }
    }
}

/// Mean distance to the k-1 nearest *other* points (k includes the point itself),
/// as in PointsTensor::scales(). Default k = 4.
func knnScales(xyz: [Float], count: Int, k: Int = 4) -> [Float] {
    var result = [Float](repeating: 0, count: count)
    guard count > 0 else { return result }
    xyz.withUnsafeBufferPointer { pts in
        let tree = PointsKDTree(points: pts, count: count)
        let kk = min(k, count)
        result.withUnsafeMutableBufferPointer { out in
            let outBase = out.baseAddress!
            let chunks = max(1, min(count, ProcessInfo.processInfo.activeProcessorCount * 8))
            let per = (count + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                var best = [Float](); best.reserveCapacity(kk)
                var stack = [Int32](); stack.reserveCapacity(64)
                let start = c * per, end = min(count, start + per)
                if start >= end { return }
                for i in start..<end {
                    let q = SIMD3(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2])
                    tree.knn(q, k: kk, into: &best, stack: &stack)
                    var sum: Float = 0
                    if best.count > 1 { for j in 1..<best.count { sum += best[j].squareRoot() } }
                    // C++ divides by (k - 1) unconditionally.
                    outBase[i] = sum / Float(k - 1)
                }
            }
        }
    }
    return result
}
