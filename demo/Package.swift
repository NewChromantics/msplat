// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "DemoApp",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(path: "../coreswift"),
		//.package(url: "https://github.com/NewChromantics/SplatScene.SwiftPackage", revision: "65932e33c921771830c27cc0e122ac880dc07c3f")
		.package(path: "/Volumes/Code/SplatScene.SwiftPackage")
    ],
    targets: [
        .executableTarget(
            name: "DemoApp",
			dependencies: [
				.product(name: "MsplatCore", package: "coreswift"),
				.product(name: "SplatScene", package: "SplatScene.SwiftPackage")
			], 
			path: "Sources",
			resources:[.copy("../../datasets/")]
        ),
    ]
)
