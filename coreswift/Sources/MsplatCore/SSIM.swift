// Port of ssim.cpp — SSIM window creation and CPU SSIM evaluation.
// Training uses the Metal SSIM kernels; this is only for metrics.

import Foundation

/// 11x11 (by default) Gaussian window, flattened row-major.
func createSSIMWindow(windowSize: Int = 11, sigma: Float = 1.5) -> [Float] {
    let g = gaussian1D(windowSize: windowSize, sigma: sigma)
    var w = [Float](repeating: 0, count: windowSize * windowSize)
    for i in 0..<windowSize {
        for j in 0..<windowSize { w[i * windowSize + j] = g[i] * g[j] }
    }
    return w
}

private func gaussian1D(windowSize: Int, sigma: Float) -> [Float] {
    var g = [Float](repeating: 0, count: windowSize)
    var sum: Float = 0
    for i in 0..<windowSize {
        let x = Float(i - windowSize / 2)
        g[i] = expf(-(x * x) / (2.0 * sigma * sigma))
        sum += g[i]
    }
    for i in 0..<windowSize { g[i] /= sum }
    return g
}

/// Separable Gaussian blur on one (H, W) channel with clamp-to-edge. `tmp` is scratch.
private func gaussianBlur(_ input: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                          _ tmp: UnsafeMutablePointer<Float>, _ H: Int, _ W: Int,
                          _ kernel: [Float]) {
    let kSize = kernel.count
    let pad = kSize / 2
    for y in 0..<H {
        for x in 0..<W {
            var sum: Float = 0
            for k in 0..<kSize {
                let sx = min(max(x + k - pad, 0), W - 1)
                sum += input[y * W + sx] * kernel[k]
            }
            tmp[y * W + x] = sum
        }
    }
    for y in 0..<H {
        for x in 0..<W {
            var sum: Float = 0
            for k in 0..<kSize {
                let sy = min(max(y + k - pad, 0), H - 1)
                sum += tmp[sy * W + x] * kernel[k]
            }
            out[y * W + x] = sum
        }
    }
}

/// CPU SSIM for (H, W, C) float images in [0, 1].
func ssimEval(_ rendered: MTensor, _ gt: MTensor, windowSize: Int = 11, sigma: Float = 1.5) -> Float {
    let H = rendered.size(0), W = rendered.size(1), C = rendered.size(2)
    let HW = H * W
    let kernel = gaussian1D(windowSize: windowSize, sigma: sigma)
    let r = rendered.floats(), g = gt.floats()

    var rCh = [Float](repeating: 0, count: HW), gCh = rCh, rr = rCh, gg = rCh, rg = rCh
    var mu1 = rCh, mu2 = rCh, sRR = rCh, sGG = rCh, sRG = rCh, tmp = rCh

    let C1: Float = 0.01 * 0.01
    let C2: Float = 0.03 * 0.03
    var ssimSum: Double = 0
    var count = 0

    for c in 0..<C {
        for i in 0..<HW {
            rCh[i] = r[i * C + c]
            gCh[i] = g[i * C + c]
            rr[i] = rCh[i] * rCh[i]
            gg[i] = gCh[i] * gCh[i]
            rg[i] = rCh[i] * gCh[i]
        }
        gaussianBlur(rCh, &mu1, &tmp, H, W, kernel)
        gaussianBlur(gCh, &mu2, &tmp, H, W, kernel)
        gaussianBlur(rr, &sRR, &tmp, H, W, kernel)
        gaussianBlur(gg, &sGG, &tmp, H, W, kernel)
        gaussianBlur(rg, &sRG, &tmp, H, W, kernel)

        for i in 0..<HW {
            let m1sq = mu1[i] * mu1[i]
            let m2sq = mu2[i] * mu2[i]
            let m12 = mu1[i] * mu2[i]
            let sig1sq = sRR[i] - m1sq
            let sig2sq = sGG[i] - m2sq
            let sig12 = sRG[i] - m12
            let num = (2.0 * m12 + C1) * (2.0 * sig12 + C2)
            let den = (m1sq + m2sq + C1) * (sig1sq + sig2sq + C2)
            ssimSum += Double(num / den)
            count += 1
        }
    }
    return Float(ssimSum / Double(count))
}

// MARK: - Metrics (model.cpp)

func psnr(_ rendered: MTensor, _ gt: MTensor) -> Float {
    let n = rendered.numel
    let r = rendered.floats(), g = gt.floats()
    var mse: Double = 0
    for i in 0..<n { let d = Double(r[i] - g[i]); mse += d * d }
    mse /= Double(n)
    return Float(10.0 * log10(1.0 / mse))
}

func l1Loss(_ rendered: MTensor, _ gt: MTensor) -> Float {
    let n = rendered.numel
    let r = rendered.floats(), g = gt.floats()
    var sum: Double = 0
    for i in 0..<n { sum += Double(abs(r[i] - g[i])) }
    return Float(sum / Double(n))
}
