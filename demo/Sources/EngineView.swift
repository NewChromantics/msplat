import SwiftUI
import MsplatCore



public struct AppError: LocalizedError
{
	public let errorDescription : String?
	
	public init(_ message: String)
	{
		errorDescription = message 
	}
}

struct ErrorWithIdentifier : Identifiable
{
	var id = UUID()
	var error : Error
}


struct PopupErrorView : View 
{
	var error : Error
	
	var body: some View 
	{
		VStack
		{
			Label("Error: \(error.localizedDescription)", systemImage: "exclamationmark.triangle.fill")
		}
		.padding(20)
	}
}

import ModelIO
import UniformTypeIdentifiers
extension UTType 
{
	static let ply = UTType(importedAs: kUTTypePolygon)
}

struct PlyFile: Transferable 
{
	let data: Data
	static var transferRepresentation: some TransferRepresentation 
	{
		DataRepresentation(exportedContentType: .ply) { $0.data }
	}
}

struct EngineStateView : View 
{
	var engine : Engine
	@State var controlError : ErrorWithIdentifier?
	var OnClickedSetSplatPose : (CameraMeta)->Void
	
	//	sits here to instigate export dialog
	@State var exportedPlyData : Data?
	var plyData : PlyFile?		{	exportedPlyData.map{ PlyFile(data: $0) }	}
	var exportingPlyData : Binding<Bool> 
	{
		Binding(
			get: { exportedPlyData != nil },
			set: { if !$0 { exportedPlyData = nil } }
		)
	}
	
	var body: some View 
	{
		VStack
		{
			RenderedImage(imageAndMeta: engine.trainerRenderedImage)
			RenderedImage(imageAndMeta: engine.trainerRenderedImage2)
			Spacer()
			EngineControls()
			EngineStats()
		}
	}
	
	@ViewBuilder func EngineStats() -> some View
	{
		VStack
		{
			HStack(spacing: 32) 
			{
				switch engine.phase 
				{
					case .loading:
						Text("Loading...")
					case .error(let error):
						Text("Error: \(error.localizedDescription)")
					case .training:
						Text("step \(engine.iteration) / \(engine.totalIterations)")
						Text("\(fmtCount(engine.splatCount)) splats")
						Text(String(format: "%.1f ms/step", engine.msPerStep))
						Text("Elapsed \(Int(engine.elapsedSecs))secs")
						Text(String(format: "GPU: %.0f MB", engine.memory.totalMB))
							.foregroundStyle(memoryColor.opacity(0.9))
					case .finished:
						Text("Finished")
				}
			}
			.monospacedDigit()
			.foregroundStyle(.white)
			.frame(maxWidth: .infinity)
			.padding(.vertical, 14)
			.background(.black.opacity(0.85))
			
			progressBar
		}
	}
	
	private var progressBar: some View 
	{
		let progress = Double(engine.iteration) / Double(engine.totalIterations)
		return GeometryReader { geo in
			ZStack(alignment: .leading) {
				Rectangle()
					.fill(.white.opacity(0.15))
				Rectangle()
					.fill(
						LinearGradient(
							colors: [Color(red: 0.2, green: 0.5, blue: 1.0),
									 Color(red: 0.0, green: 0.9, blue: 0.7)],
							startPoint: .leading, endPoint: .trailing
						)
					)
					.frame(width: geo.size.width * progress)
					.animation(.linear(duration: 0.1), value: progress)
			}
		}
		.frame(height: 6)
	}
	
	
	private func fmtCount(_ n: Int) -> String {
		if n >= 1_000_000 { return String(format: "%.2fM", Double(n) / 1_000_000) }
		if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1_000) }
		return "\(n)"
	}
	
	private var memoryColor: Color {
		if engine.memory.totalMB > 4000 {
			return .red
		} else if engine.memory.totalMB > 2000 {
			return .orange
		} else {
			return Color(red: 0.0, green: 0.9, blue: 0.7)
		}
	}
	
	private var efficiencyColor: Color {
		if engine.memory.efficiency < 15 {
			return .red
		} else if engine.memory.efficiency < 25 {
			return .orange
		} else {
			return .green
		}
	}
	
	private var memoryBreakdownText: String
	{
		let kbPerSplat = engine.memory.mbPerSplat * 1024.0
		return String(format: "Model: %.1f MB\nGPU Cache: %.1f MB\nCPU Images: %.1f MB\n%.1f KB/splat", 
					  engine.memory.modelMB, engine.memory.datasetMB, engine.memory.cpuImagesMB, kbPerSplat)
	}

	@ViewBuilder func EngineControls() -> some View
	{
		HStack
		{
			Button(action:OnClickedExportSplatsPly)
			{
				Label("Export .ply", systemImage: "arrow.down.document.fill")
			}
		}
		.popover(item: $controlError)
		{
			PopupErrorView(error:$0.error)
		}
		.fileExporter(isPresented: exportingPlyData,item: plyData,defaultFilename: "export")
		{
			switch $0
			{
				case .failure(let error):	controlError = ErrorWithIdentifier(error: error)
				default:
					break
			}
		}
	}

	@ViewBuilder func RenderedImage(imageAndMeta:(image:CGImage,meta:CameraMeta)?) -> some View
	{
		if let imageAndMeta
		{
			RenderedImage(image: imageAndMeta.image, cameraMeta: imageAndMeta.meta)
		}
	}
	
	
	@ViewBuilder func RenderedImage(image:CGImage,cameraMeta:CameraMeta) -> some View
	{
		Image(nsImage: NSImage(cgImage: image, size: image.size))
			.resizable()
			.aspectRatio(contentMode: .fit)
			.frame(minWidth: 300,maxWidth: 500,maxHeight: .infinity)
			.overlay(alignment:.topTrailing)
		{
			VStack
			{
				Button(action: {OnClickedSetSplatPose(cameraMeta)})
				{
					Label("Copy camera pose", systemImage: "arrow.right.square.fill")
						.padding(5)
				}
			}
			.controlSize(.mini)
			.darkBackground()
		}
	}

	func OnClickedExportSplatsPly()
	{
		RunAsync
		{
			self.exportedPlyData = try await engine.splats.ExportPly(transform: .identity)
		}
	}
	
	func RunAsync(functor:@escaping() async throws->Void)
	{
		Task{ @MainActor in
			do
			{
				try await functor()
			}
			catch
			{
				self.controlError = .init(error: error)
			}
		}
	}
	
}
