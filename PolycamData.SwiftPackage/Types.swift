import Foundation


public struct NerfStudioError: LocalizedError 
{
	public var errorDescription: String?
	
	public init(_ description: String) 
	{
		self.errorDescription = description
	}
}




/// Represents a camera from a Nerfstudio dataset (transforms.json)
public struct NerfStudioCamera {
    public var width: Int
    public var height: Int
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float
    public var k1: Float
    public var k2: Float
    public var k3: Float
    public var k4: Float
    public var p1: Float
    public var p2: Float
    /// 4x4 row-major camera-to-world transform (OpenGL convention: Y-up, Z-back)
    public var transformMatrix: [Float]
    /// Path to the image file, relative to the dataset directory
    public var imagePath: String
    
    public init(
        width: Int = 0,
        height: Int = 0,
        fx: Float = 0,
        fy: Float = 0,
        cx: Float = 0,
        cy: Float = 0,
        k1: Float = 0,
        k2: Float = 0,
        k3: Float = 0,
        k4: Float = 0,
        p1: Float = 0,
        p2: Float = 0,
        transformMatrix: [Float] = Array(repeating: 0, count: 16),
        imagePath: String = ""
    ) {
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.p1 = p1
        self.p2 = p2
        self.transformMatrix = transformMatrix
        self.imagePath = imagePath
    }
}


/// Represents optional point cloud data in a Polycam dataset
public struct PolycamPointCloud {
	/// XYZ positions (count * 3 floats)
	public var positions: [Float]
	/// RGB colors (count * 3 bytes)
	public var colors: [UInt8]
	/// Number of points
	public var count: Int
	
	public init(positions: [Float], colors: [UInt8], count: Int) {
		self.positions = positions
		self.colors = colors
		self.count = count
	}
}


