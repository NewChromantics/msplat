// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "DemoApp",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../coreswift"),
    ],
    targets: [
        .executableTarget(
            name: "DemoApp",
			dependencies: [
				.product(name: "MsplatCore", package: "coreswift"),
			], 
			path: "Sources",
			resources:[.copy("../../datasets/")]
        ),
    ]
)
