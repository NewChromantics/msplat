// Bit-exact reimplementations of the libc++ random facilities used by the C++
// core, so the Swift port draws the same random numbers as the C++ build:
//   std::mt19937, uniform_real_distribution<float>, normal_distribution<float>,
//   uniform_int_distribution<ptrdiff_t> (via std::shuffle).

import Foundation

/// std::mt19937 (32-bit Mersenne Twister).
struct MT19937 {
    private var mt = [UInt32](repeating: 0, count: 624)
    private var index = 624

    static let min: UInt32 = 0
    static let max: UInt32 = .max

    init(seed: UInt32 = 5489) {
        mt[0] = seed
        for i in 1..<624 {
            mt[i] = 1_812_433_253 &* (mt[i - 1] ^ (mt[i - 1] >> 30)) &+ UInt32(i)
        }
        index = 624
    }

    private mutating func twist() {
        for i in 0..<624 {
            let y = (mt[i] & 0x8000_0000) | (mt[(i + 1) % 624] & 0x7FFF_FFFF)
            var v = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1 != 0 { v ^= 0x9908_B0DF }
            mt[i] = v
        }
        index = 0
    }

    mutating func next() -> UInt32 {
        if index >= 624 { twist() }
        var y = mt[index]
        index += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= (y >> 18)
        return y
    }
}

/// libc++ generate_canonical<float, 24>(mt19937): one draw, computed in float.
@inline(__always)
func generateCanonicalFloat(_ g: inout MT19937) -> Float {
    let r: Float = Float(UInt32.max) + 1.0  // 2^32 (exact in float)
    let s = Float(g.next() - MT19937.min)
    return s / r
}

/// libc++ uniform_real_distribution<float>(a, b).
struct UniformRealFloat {
    let a: Float, b: Float
    init(_ a: Float, _ b: Float) { self.a = a; self.b = b }
    func sample(_ g: inout MT19937) -> Float {
        (b - a) * generateCanonicalFloat(&g) + a
    }
}

/// libc++ normal_distribution<float> (Marsaglia polar method, caches 2nd value).
struct NormalFloat {
    let mean: Float, stddev: Float
    private var cached: Float = 0
    private var hot = false

    init(mean: Float = 0, stddev: Float = 1) { self.mean = mean; self.stddev = stddev }

    mutating func sample(_ g: inout MT19937) -> Float {
        var up: Float
        if hot {
            hot = false
            up = cached
        } else {
            let uni = UniformRealFloat(-1, 1)
            var u: Float, v: Float, s: Float
            repeat {
                u = uni.sample(&g)
                v = uni.sample(&g)
                s = u * u + v * v
            } while s > 1 || s == 0
            let f = (-2 * logf(s) / s).squareRoot()
            cached = v * f
            hot = true
            up = u * f
        }
        return up * stddev + mean
    }
}

/// libc++ uniform_int_distribution<ptrdiff_t>(0, b) driven by mt19937.
/// Implements __independent_bits_engine for widths up to 32 bits (always the case
/// here: b is a camera count).
func uniformIntLibcxx(_ g: inout MT19937, upTo b: Int) -> Int {
    precondition(b >= 0)
    let rp = UInt64(b) + 1
    if rp == 1 { return 0 }
    let dt = 64
    var w = dt - rp.leadingZeroBitCount - 1
    if (rp & (UInt64.max >> UInt64(dt - w))) != 0 { w += 1 }
    precondition(w <= 32, "uniformIntLibcxx: ranges wider than 32 bits are not supported")
    // n == 1, w0 == w, y0 == 2^32 → no rejection on the engine draw itself.
    let mask: UInt32 = w >= 32 ? .max : (UInt32.max >> UInt32(32 - w))
    var u: UInt64
    repeat {
        u = UInt64((g.next() - MT19937.min) & mask)
    } while u >= rp
    return Int(u)
}

/// std::shuffle(first, last, mt19937) as implemented by libc++.
func shuffleLibcxx<T>(_ a: inout [T], _ g: inout MT19937) {
    guard a.count > 1 else { return }
    // for (--last, --d; first < last; ++first, --d)
    var first = 0
    let last = a.count - 1
    var d = a.count - 1
    while first < last {
        let i = uniformIntLibcxx(&g, upTo: d)
        if i != 0 { a.swapAt(first, first + i) }
        first += 1
        d -= 1
    }
}
