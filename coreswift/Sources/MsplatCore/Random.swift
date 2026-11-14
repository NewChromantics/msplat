// Fast deterministic random number generators for training reproducibility.
// Xoshiro256++ is ~5x faster than MT19937 with better statistical properties.
//
// USAGE:
//   var rng = Xoshiro256PlusPlus(seed: 42)  // Or MT19937(seed: 42) for old behavior
//   let uniform = UniformRealFloat(0, 1)
//   let value = uniform.sample(&rng)
//
// To switch RNG globally, search/replace Xoshiro256PlusPlus <-> MT19937

import Foundation

/// Xoshiro256++ - Fast, high-quality 64-bit random number generator.
/// Roughly 5x faster than MT19937 with better statistical properties.
/// See: https://prng.di.unimi.it/
struct Xoshiro256PlusPlus: RandomNumberGenerator {
    private var s0: UInt64
    private var s1: UInt64
    private var s2: UInt64
    private var s3: UInt64
    
    static let min: UInt32 = 0
    static let max: UInt32 = .max
    
    init(seed: UInt64 = 0x123456789ABCDEF) {
        // SplitMix64 to initialize state from single seed
        var z = seed &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        s0 = z ^ (z >> 31)
        
        z = s0 &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        s1 = z ^ (z >> 31)
        
        z = s1 &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        s2 = z ^ (z >> 31)
        
        z = s2 &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        s3 = z ^ (z >> 31)
    }
    
    @inline(__always)
    private static func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
    
    mutating func next() -> UInt64 {
        let result = Self.rotl(s0 &+ s3, 23) &+ s0
        let t = s1 << 17
        
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = Self.rotl(s3, 45)
        
        return result
    }
    
    // Generate UInt32 for compatibility with existing code
    mutating func next32() -> UInt32 {
        UInt32(truncatingIfNeeded: next())
    }
}

// Typealias for easy swapping - change this to switch RNG implementations
typealias FastRNG = Xoshiro256PlusPlus

/// std::mt19937 (32-bit Mersenne Twister) - DEPRECATED: Use FastRNG instead.
/// Kept for reference but ~5x slower than Xoshiro256++.
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

// MARK: - Protocol for our RNGs

protocol FastRandomNumberGenerator {
    mutating func next32() -> UInt32
    static var min: UInt32 { get }
    static var max: UInt32 { get }
}

extension Xoshiro256PlusPlus: FastRandomNumberGenerator {}

extension MT19937: FastRandomNumberGenerator {
    mutating func next32() -> UInt32 {
        return next()
    }
}

// MARK: - Distribution Functions

/// Generate canonical float [0, 1) from any FastRandomNumberGenerator.
@inline(__always)
func generateCanonicalFloat<G: FastRandomNumberGenerator>(_ g: inout G) -> Float {
    let r: Float = Float(UInt32.max) + 1.0  // 2^32 (exact in float)
    let s = Float(g.next32() - G.min)
    return s / r
}

/// Uniform real distribution over [a, b).
struct UniformRealFloat {
    let a: Float, b: Float
    init(_ a: Float, _ b: Float) { self.a = a; self.b = b }
    
    func sample<G: FastRandomNumberGenerator>(_ g: inout G) -> Float {
        (b - a) * generateCanonicalFloat(&g) + a
    }
}

/// Normal (Gaussian) distribution (Marsaglia polar method, caches 2nd value).
struct NormalFloat {
    let mean: Float, stddev: Float
    private var cached: Float = 0
    private var hot = false

    init(mean: Float = 0, stddev: Float = 1) { self.mean = mean; self.stddev = stddev }

    mutating func sample<G: FastRandomNumberGenerator>(_ g: inout G) -> Float {
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

/// Uniform integer distribution [0, b].
func uniformInt<G: FastRandomNumberGenerator>(_ g: inout G, upTo b: Int) -> Int {
    precondition(b >= 0)
    let rp = UInt64(b) + 1
    if rp == 1 { return 0 }
    let dt = 64
    var w = dt - rp.leadingZeroBitCount - 1
    if (rp & (UInt64.max >> UInt64(dt - w))) != 0 { w += 1 }
    precondition(w <= 32, "uniformInt: ranges wider than 32 bits are not supported")
    let mask: UInt32 = w >= 32 ? .max : (UInt32.max >> UInt32(32 - w))
    var u: UInt64
    repeat {
        u = UInt64((g.next32() - G.min) & mask)
    } while u >= rp
    return Int(u)
}

/// Shuffle array using Fisher-Yates algorithm.
func shuffle<T, G: FastRandomNumberGenerator>(_ a: inout [T], _ g: inout G) {
    guard a.count > 1 else { return }
    var first = 0
    let last = a.count - 1
    var d = a.count - 1
    while first < last {
        let i = uniformInt(&g, upTo: d)
        if i != 0 { a.swapAt(first, first + i) }
        first += 1
        d -= 1
    }
}

// MARK: - Deprecated libc++ compatibility functions

/// libc++ uniform_int_distribution<ptrdiff_t>(0, b) - DEPRECATED: Use uniformInt instead.
@available(*, deprecated, renamed: "uniformInt")
func uniformIntLibcxx(_ g: inout MT19937, upTo b: Int) -> Int {
    uniformInt(&g, upTo: b)
}

/// std::shuffle - DEPRECATED: Use shuffle instead.
@available(*, deprecated, renamed: "shuffle")
func shuffleLibcxx<T>(_ a: inout [T], _ g: inout MT19937) {
    shuffle(&a, &g)
}
