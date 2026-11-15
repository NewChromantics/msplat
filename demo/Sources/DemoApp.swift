import SwiftUI
import AppKit



class AppDelegate: NSObject, NSApplicationDelegate 
{
	func applicationDidFinishLaunching(_ notification: Notification) 
	{
		NSApp.setActivationPolicy(.regular)
		NSApp.activate(ignoringOtherApps: true)
	}
}



@main
struct DemoApp: App
{
	@NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
	let datasetPath: URL
	
	init() {
		let args = CommandLine.arguments
		
		if let url = Bundle.module.url(forResource: "datasets/mipnerf360/garden", withExtension: nil) 
		{
			datasetPath = url  // Convert URL to path string
		}
		else
		{
			let path = args.count > 1 ? args[1] : "datasets/mipnerf360/garden"
			datasetPath = URL(filePath: path)
		}
	}
	
	var body: some Scene 
	{
		WindowGroup {
			ContentView(datasetPath: datasetPath)
				.frame(minWidth: 200, minHeight: 200)
		}
		.defaultSize(width: 1280, height: 720)
	}
}
