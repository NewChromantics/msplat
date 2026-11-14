import Foundation


public struct ColmapCamera 
{
	public var id: UInt32 = 0
	public var model = 0
	public var width = 0, height = 0
	public var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
	public var k1: Float = 0, k2: Float = 0, k3: Float = 0, p1: Float = 0, p2: Float = 0
}

public struct ColmapImage 
{
	public var camId: UInt32 = 0
	public var quat: [Double] = []  // w, x, y, z
	public var t: [Double] = []     // world-to-camera translation
	public var filename = ""
}


public struct ColmapError: LocalizedError 
{
	public var errorDescription: String?
	
	public init(_ description: String) 
	{
		self.errorDescription = description
	}
}



enum ColmapCameraModel: UInt32 {
	case simplePinhole = 0
	case pinhole = 1
	case simpleRadial = 2
	case radial = 3
	case opencv = 4
	case fullOpencv = 5
	case simpleRadialFisheye = 6
	case radialFisheye = 7
	case thinPrismFisheye = 12
}



public struct ColmapCameraWithTransform {
	public var id: Int
	public var width: Int
	public var height: Int
	public var fx: Float
	public var fy: Float
	public var cx: Float
	public var cy: Float
	public var k1: Float
	public var k2: Float
	public var k3: Float
	public var p1: Float
	public var p2: Float
	/// 4x4 row-major camera-to-world transform matrix (OpenGL convention: Y-up, Z-back)
	public var transformMatrix: [Float]
	/// Relative path to the image file
	public var imagePath: String
	
	public init(
		id: Int,
		width: Int,
		height: Int,
		fx: Float,
		fy: Float,
		cx: Float,
		cy: Float,
		k1: Float = 0,
		k2: Float = 0,
		k3: Float = 0,
		p1: Float = 0,
		p2: Float = 0,
		transformMatrix: [Float],
		imagePath: String
	) {
		self.id = id
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.k1 = k1
		self.k2 = k2
		self.k3 = k3
		self.p1 = p1
		self.p2 = p2
		self.transformMatrix = transformMatrix
		self.imagePath = imagePath
	}
}
