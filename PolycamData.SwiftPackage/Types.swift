import Foundation


// Represents optional point cloud data in a Polycam dataset
public struct PolycamPointCloud
{
	// XYZ positions (count * 3 floats)
	public var positions: [Float]
	// RGB colors (count * 3 bytes)
	public var colors: [UInt8]
	// Number of points
	public var count: Int
	
	public init(positions: [Float], colors: [UInt8], count: Int) {
		self.positions = positions
		self.colors = colors
		self.count = count
	}
}


