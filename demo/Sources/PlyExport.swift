import Foundation
import MsplatCore
import PopPly
import simd


extension simd_float3
{
	var floats : [Float]	{	[x,y,z]	}
}
extension simd_float4
{
	var floats : [Float]	{	[x,y,z,w]	}
}
extension simd_quatf
{
	var wxyzFloats : [Float]	
	{
		[self.vector.w,self.vector.x,self.vector.y,self.vector.z]
	}
}

extension Collection where Element == PlySplatPoint
{
	func ExportPly(transform:simd_float4x4) async throws -> Data
	{
		//	todo: allow an empty ply, but need DC meta etc
		guard let firstSplat = self.first else
		{
			throw PlyError("No splats to export")
		}
		
		let numDc = firstSplat.dcCount	//p.featuresDc.size(1)
		let frBases = firstSplat.featureRestBases	// p.featuresRest.size(-2)
		let numFr = frBases * 3
		let splatCount = self.count
		
		var header = "ply\nformat binary_little_endian 1.0\n"
		header += "element vertex \(splatCount)\n"
		header += "property float x\nproperty float y\nproperty float z\n"
		header += "property float nx\nproperty float ny\nproperty float nz\n"
		for i in 0..<numDc { header += "property float f_dc_\(i)\n" }
		for i in 0..<numFr { header += "property float f_rest_\(i)\n" }
		header += "property float opacity\n"
		header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
		header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
		header += "end_header\n"
		
		func SplatAsFloats(splat:PlySplatPoint) -> [Float]
		{
			var floats : [Float] = []
			
			let pos = (transform * splat.position.xyz1).xyz
			//out[c] = p.keepCrs ? (mp[i * 3 + j] / p.scale + tr[j]) : mp[i * 3 + j]
			let normal = simd_float3.zero
			let scale = (transform * splat.scale.xyz0).xyz
			let scaleLog = scale.floats.map(logf)
			let featureRests : [Float] = []
			let rotationWxyz = splat.quaternionWxyz
			
			floats.append(contentsOf:pos.floats)
			floats.append(contentsOf:normal.floats)
			floats.append(contentsOf:splat.shDc.floats)
			
			/*
			 // Transpose [frBases, 3] → [3, frBases] for PLY convention
			 for ch in 0..<3 
			 {
			 for b in 0..<frBases 
			 {
			 out[c] = frp[i * frBases * 3 + b * 3 + ch] 
			 c += 1 
			 }
			 }
			 */
			floats.append(contentsOf:featureRests)
			
			floats.append(splat.opacityLogit)
			floats.append(contentsOf: scaleLog)
			floats.append(contentsOf: rotationWxyz.floats)
			
			return floats
		}
		
		//let floatRows = [SplatAsFloats(splat: firstSplat)]
		let floatRows = self.map(SplatAsFloats)
		let body = floatRows.flatMap{ $0 }
		
		//	encode floats to string
		var data = Data(header.utf8)
		body.withUnsafeBytes 
		{
			data.append(contentsOf: $0) 
		}
		return data
	}
}
