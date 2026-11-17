// Port of loaders/image_io.cpp: CoreGraphics image read/write, area resize and
// Brown-Conrady undistortion.

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreGraphicsExtensions



extension CGImage {
    /// Export RGB float data with values in the range [0, 1].
    /// - Returns: A tuple containing the float array (HWC format), width, and height
    func exportRGBFloat() -> (data: [Float], width: Int, height: Int) {
        let w = width
        let h = height
        
        // Create RGBA buffer
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        rgba.withUnsafeMutableBytes { buf in
            let context = CGContext(
                data: buf.baseAddress,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: w * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )!
            context.draw(self, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        
        // Convert to RGB float [0, 1]
        var pixels = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            pixels[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
            pixels[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
            pixels[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
        }
        
        return (data: pixels, width: w, height: h)
    }
	
	func exportRGBAFloat() -> (data: [Float], width: Int, height: Int) {
		let w = width
		let h = height
		
		// Create RGBA buffer
		var rgba = [UInt8](repeating: 0, count: w * h * 4)
		let colorSpace = CGColorSpaceCreateDeviceRGB()
		
		rgba.withUnsafeMutableBytes { buf in
			let context = CGContext(
				data: buf.baseAddress,
				width: w,
				height: h,
				bitsPerComponent: 8,
				bytesPerRow: w * 4,
				space: colorSpace,
				bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
			)!
			context.draw(self, in: CGRect(x: 0, y: 0, width: w, height: h))
		}
		
		// Convert to RGB float [0, 1]
		var pixels = [Float](repeating: 0, count: w * h * 4)
		for i in 0..<(w * h) {
			pixels[i * 4 + 0] = Float(rgba[i * 4 + 0]) / 255.0
			pixels[i * 4 + 1] = Float(rgba[i * 4 + 1]) / 255.0
			pixels[i * 4 + 2] = Float(rgba[i * 4 + 2]) / 255.0
			pixels[i * 4 + 3] = Float(rgba[i * 4 + 3]) / 255.0
		}
		
		return (data: pixels, width: w, height: h)
	}
    
    /// Export as RGBImage structure.
    /// - Returns: An RGBImage with float RGB values in [0, 1]
    /// - Throws: If the image dimensions are invalid
    func exportRGBImage() throws -> RGBImage 
	{
        let (data, width, height) = exportRGBFloat()
		return try RGBImage(data: data, width: width, height: height, format: .rgb)
    }
	
	func exportRGBAImage() throws -> RGBImage 
	{
		let (data, width, height) = exportRGBAFloat()
		return try RGBImage(data: data, width: width, height: height, format: .rgba)
	}
}

/// Float32 RGB image, HWC, values in [0, 1].
public struct RGBImage 
{
	public enum ImageFormat
	{
		case rgb
		case rgba
		
		public var channels : Int
		{
			switch self
			{
				case .rgb:	3
				case .rgba:	4
			}
		}
	}
    public var data: [Float]
	public var width: Int
	public var height: Int
	public var format : ImageFormat
	public var channels : Int		{	format.channels	}
    
	public init(data: [Float], width: Int, height: Int,format:ImageFormat) throws {
        guard width > 0 && height > 0 else {
            throw MsplatError("RGBImage dimensions must be positive: \(width)x\(height)")
        }
		let expectedSize = width * height * format.channels
        guard data.count == expectedSize else {
            throw MsplatError("RGBImage data size mismatch: expected \(expectedSize) (3×\(width)×\(height)), got \(data.count)")
        }
        self.data = data
        self.width = width
        self.height = height
		self.format = format
    }
    
    /// Load an RGB image from a file URL
	public init(contentsOf url: URL) throws 
	{
		guard let cg = CGImage.load(contentsOf: url) else
		{
            throw MsplatError("Failed to load image: \(url.path)")
        }
		let rgb = cg.exportRGBFloat()

		try self.init(data: rgb.data, width: rgb.width, height: rgb.height, format: .rgb)
    }
	
	/// Create a CGImage from this RGBImage
	/// - Parameter useFloatComponents: If true, uses 32-bit float components (faster, zero-copy for RGBA). If false, converts to 8-bit (more compatible).
	/// - Returns: A CGImage 
	/// - Throws: If CGDataProvider or CGImage creation fails
	public func makeCGImage(useFloatComponents: Bool = false) throws -> CGImage {
		if useFloatComponents {
			// Fast path: Use float components directly (values 0-1, interpreted as floats)
			let componentsPerPixel = 4
			let bytesPerComponent = MemoryLayout<Float>.size
			let bytesPerRow = width * componentsPerPixel * bytesPerComponent
			guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
				throw MsplatError("Failed to create sRGB color space")
			}
			let bitmapInfo = CGBitmapInfo(rawValue:
				CGImageAlphaInfo.premultipliedLast.rawValue |
				CGBitmapInfo.floatComponents.rawValue |
				CGBitmapInfo.byteOrder32Little.rawValue)
			
			if channels == 4 {
				// Zero-copy path for RGBA
				guard let provider = CGDataProvider(data: Data(bytes: data, count: data.count * bytesPerComponent) as CFData) else {
					throw MsplatError("Failed to create CGDataProvider")
				}
				
				guard let cgImage = CGImage(
					width: width,
					height: height,
					bitsPerComponent: 32,
					bitsPerPixel: componentsPerPixel * 32,
					bytesPerRow: bytesPerRow,
					space: colorSpace,
					bitmapInfo: bitmapInfo,
					provider: provider,
					decode: nil,
					shouldInterpolate: true,
					intent: .defaultIntent
				) else {
					throw MsplatError("Failed to create CGImage")
				}
				return cgImage
			} else {
				// RGB -> RGBA with float components
				var rgba = [Float](repeating: 0, count: width * height * componentsPerPixel)
				for i in 0..<(width * height) {
					rgba[i * 4 + 0] = data[i * 3 + 0]
					rgba[i * 4 + 1] = data[i * 3 + 1]
					rgba[i * 4 + 2] = data[i * 3 + 2]
					rgba[i * 4 + 3] = 1.0
				}
				
				guard let provider = CGDataProvider(data: Data(bytes: rgba, count: rgba.count * bytesPerComponent) as CFData) else {
					throw MsplatError("Failed to create CGDataProvider")
				}
				
				guard let cgImage = CGImage(
					width: width,
					height: height,
					bitsPerComponent: 32,
					bitsPerPixel: componentsPerPixel * 32,
					bytesPerRow: bytesPerRow,
					space: colorSpace,
					bitmapInfo: bitmapInfo,
					provider: provider,
					decode: nil,
					shouldInterpolate: true,
					intent: .defaultIntent
				) else {
					throw MsplatError("Failed to create CGImage")
				}
				return cgImage
			}
		} else {
			// Standard path: Convert to 8-bit RGBA (most compatible)
			let n = width * height
			let rgba = UnsafeMutablePointer<UInt8>.allocate(capacity: n * 4)
			
			// Convert float RGB/RGBA to 8-bit RGBA
			data.withUnsafeBufferPointer { src in
				if channels == 3 {
					// RGB -> RGBA
					for i in 0..<n {
						rgba[i * 4 + 0] = UInt8(min(max(src[i * 3 + 0], 0), 1) * 255)
						rgba[i * 4 + 1] = UInt8(min(max(src[i * 3 + 1], 0), 1) * 255)
						rgba[i * 4 + 2] = UInt8(min(max(src[i * 3 + 2], 0), 1) * 255)
						rgba[i * 4 + 3] = 255
					}
				} else if channels == 4 {
					// RGBA -> RGBA
					for i in 0..<n {
						rgba[i * 4 + 0] = UInt8(min(max(src[i * 4 + 0], 0), 1) * 255)
						rgba[i * 4 + 1] = UInt8(min(max(src[i * 4 + 1], 0), 1) * 255)
						rgba[i * 4 + 2] = UInt8(min(max(src[i * 4 + 2], 0), 1) * 255)
						rgba[i * 4 + 3] = UInt8(min(max(src[i * 4 + 3], 0), 1) * 255)
					}
				}
			}
			
			let imageData = Data(bytesNoCopy: rgba, count: n * 4, deallocator: .custom { ptr, _ in
				ptr.deallocate()
			})
			
			guard let provider = CGDataProvider(data: imageData as CFData) else {
				throw MsplatError("Failed to create CGDataProvider")
			}
			
			guard let cgImage = CGImage(
				width: width,
				height: height,
				bitsPerComponent: 8,
				bitsPerPixel: 32,
				bytesPerRow: width * 4,
				space: CGColorSpaceCreateDeviceRGB(),
				bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
				provider: provider,
				decode: nil,
				shouldInterpolate: false,
				intent: .defaultIntent
			) else {
				throw MsplatError("Failed to create CGImage")
			}
			
			return cgImage
		}
	}
}

/// Area-based (box filter) resize.
func resizeArea(_ src: RGBImage, _ dstW: Int, _ dstH: Int) -> RGBImage 
{
	let pixels = [Float](repeating: 0, count: dstW * dstH * src.channels)
	var dst = try! RGBImage(data: pixels, width: dstW, height: dstH, format: src.format)
    let scaleX = Float(src.width) / Float(dstW)
    let scaleY = Float(src.height) / Float(dstH)

	let outputFormat = src.format
    src.data.withUnsafeBufferPointer { s in
        dst.data.withUnsafeMutableBufferPointer { d in
            for dy in 0..<dstH {
                let srcY0 = Float(dy) * scaleY
                let srcY1 = Float(dy + 1) * scaleY
                for dx in 0..<dstW {
                    let srcX0 = Float(dx) * scaleX
                    let srcX1 = Float(dx + 1) * scaleX
                    var s0: Float = 0, s1: Float = 0, s2: Float = 0
                    var totalArea: Float = 0

                    let iy0 = Int(srcY0)
                    let iy1 = min(Int(srcY1.rounded(.up)), src.height)
                    let ix0 = Int(srcX0)
                    let ix1 = min(Int(srcX1.rounded(.up)), src.width)

                    if iy0 < iy1 {
                        for iy in iy0..<iy1 {
                            let wy = min(Float(iy + 1), srcY1) - max(Float(iy), srcY0)
                            if ix0 < ix1 {
                                for ix in ix0..<ix1 {
                                    let wx = min(Float(ix + 1), srcX1) - max(Float(ix), srcX0)
                                    let area = wx * wy
                                    let o = (iy * src.width + ix) * outputFormat.channels
                                    s0 += s[o] * area
                                    s1 += s[o + 1] * area
                                    s2 += s[o + 2] * area
                                    totalArea += area
                                }
                            }
                        }
                    }
					let o = (dy * dstW + dx) * outputFormat.channels
                    let inv = 1.0 / totalArea
                    d[o] = s0 * inv
                    d[o + 1] = s1 * inv
                    d[o + 2] = s2 * inv
                }
            }
        }
    }
    return dst
}

// MARK: - Undistortion (Brown-Conrady, alpha = 0 crop)

struct UndistortResult {
    var image: RGBImage
    var fx: Float, fy: Float, cx: Float, cy: Float
    var width: Int, height: Int
}

/// Forward distortion: normalized undistorted → normalized distorted.
@inline(__always)
private func distortPoint(_ x: Float, _ y: Float, _ k1: Float, _ k2: Float,
                          _ p1: Float, _ p2: Float, _ k3: Float) -> (Float, Float) {
    let r2 = x * x + y * y
    let r4 = r2 * r2
    let r6 = r4 * r2
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return (xd, yd)
}

/// Iterative inverse distortion: normalized distorted → normalized undistorted.
@inline(__always)
private func undistortPoint(_ xd: Float, _ yd: Float, _ k1: Float, _ k2: Float,
                            _ p1: Float, _ p2: Float, _ k3: Float) -> (Float, Float) {
    var xu = xd, yu = yd
    for _ in 0..<20 {
        let r2 = xu * xu + yu * yu
        let r4 = r2 * r2
        let r6 = r4 * r2
        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        let dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        let dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        xu = (xd - dx) / radial
        yu = (yd - dy) / radial
    }
    return (xu, yu)
}

@inline(__always)
private func bilinearSample(_ img: UnsafeBufferPointer<Float>, _ w: Int, _ h: Int,
                            _ x: Float, _ y: Float, _ out: UnsafeMutableBufferPointer<Float>, _ o: Int) {
    let fxl = x.rounded(.down), fyl = y.rounded(.down)
    var x0 = Int(fxl), y0 = Int(fyl)
    var x1 = x0 + 1, y1 = y0 + 1
    x0 = min(max(x0, 0), w - 1); x1 = min(max(x1, 0), w - 1)
    y0 = min(max(y0, 0), h - 1); y1 = min(max(y1, 0), h - 1)
    let fx = x - fxl, fy = y - fyl
    let p00 = (y0 * w + x0) * 3, p10 = (y0 * w + x1) * 3
    let p01 = (y1 * w + x0) * 3, p11 = (y1 * w + x1) * 3
    for c in 0..<3 {
        let top = img[p00 + c] * (1.0 - fx) + img[p10 + c] * fx
        let bottom = img[p01 + c] * (1.0 - fx) + img[p11 + c] * fx
        out[o + c] = top * (1.0 - fy) + bottom * fy
    }
}

func undistortImage(_ src: RGBImage, fx: Float, fy: Float, cx: Float, cy: Float,
                    k1: Float, k2: Float, p1: Float, p2: Float, k3: Float) -> UndistortResult {
    let w = src.width, h = src.height
    let nSamples = 200

    // Inner rectangle of the undistorted boundary (alpha = 0: no black borders).
    // (The C++ also computes an unused overall min/max bbox first; omitted.)
    var topMax: Float = -1e9, bottomMin: Float = 1e9, leftMax: Float = -1e9, rightMin: Float = 1e9
    for i in 0..<nSamples {
        let t = Float(i) / Float(nSamples - 1)
        // Top edge (y = 0)
        var (xu, yu) = undistortPoint((t * Float(w) - cx) / fx, (0.0 - cy) / fy, k1, k2, p1, p2, k3)
        topMax = max(topMax, yu * fy + cy)
        // Bottom edge (y = h-1)
        (xu, yu) = undistortPoint((t * Float(w) - cx) / fx, (Float(h - 1) - cy) / fy, k1, k2, p1, p2, k3)
        bottomMin = min(bottomMin, yu * fy + cy)
        // Left edge (x = 0)
        (xu, yu) = undistortPoint((0.0 - cx) / fx, (t * Float(h) - cy) / fy, k1, k2, p1, p2, k3)
        leftMax = max(leftMax, xu * fx + cx)
        // Right edge (x = w-1)
        (xu, yu) = undistortPoint((Float(w - 1) - cx) / fx, (t * Float(h) - cy) / fy, k1, k2, p1, p2, k3)
        rightMin = min(rightMin, xu * fx + cx)
    }

    var roiX = max(0, Int(leftMax.rounded(.up)))
    var roiY = max(0, Int(topMax.rounded(.up)))
    var roiW = min(w, Int(rightMin.rounded(.down))) - roiX
    var roiH = min(h, Int(bottomMin.rounded(.down))) - roiY
    if roiW <= 0 || roiH <= 0 { roiX = 0; roiY = 0; roiW = w; roiH = h }

    // For each output pixel, forward-distort to find the source pixel. Only the ROI
    // is kept, so only the ROI is computed (same result as computing then cropping).
	let format = RGBImage.ImageFormat.rgb
	let pixels = [Float](repeating: 0, count: roiW * roiH * format.channels)
	var cropped = try! RGBImage(data: pixels, width: roiW, height: roiH, format: format)
    src.data.withUnsafeBufferPointer { s in
        cropped.data.withUnsafeMutableBufferPointer { d in
            for y in 0..<roiH {
                let oy = y + roiY
                for x in 0..<roiW {
                    let ox = x + roiX
                    let (xd, yd) = distortPoint((Float(ox) - cx) / fx, (Float(oy) - cy) / fy, k1, k2, p1, p2, k3)
                    bilinearSample(s, w, h, xd * fx + cx, yd * fy + cy, d, (y * roiW + x) * format.channels)
                }
            }
        }
    }

    return UndistortResult(image: cropped, fx: fx, fy: fy,
                           cx: cx - Float(roiX), cy: cy - Float(roiY),
                           width: roiW, height: roiH)
}
