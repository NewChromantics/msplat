// swift-tools-version: 6.0
import PackageDescription

// Pure-Swift port of core/src (C++/ObjC++). The Metal shader is unchanged and
// is shipped as a resource; see MetalContext.swift for how it is loaded.
let package = Package(
    name: "MsplatCore",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MsplatCore", targets: ["MsplatCore"]),
    ],
	dependencies: 
		[
			.package(name: "NerfStudioData", path: "../NerfStudioData.SwiftPackage"),
			.package(name: "ColmapData", path: "../ColmapData.SwiftPackage"),
			.package(name: "PolycamData", path: "../PolycamData.SwiftPackage"),
			.package(url: "https://github.com/NewChromantics/CoreGraphicsExtensions.SwiftPackage", branch:"main"),
			.package(url: "https://github.com/NewChromantics/PopCommon.SwiftPackage", branch:"main")
		]
	,
    targets: [
        .target(
            name: "MsplatCore",
            dependencies: [
				"NerfStudioData",
				"ColmapData",
				"PolycamData",
				.product(name: "CoreGraphicsExtensions", package: "CoreGraphicsExtensions.SwiftPackage"),
				.product(name: "PopCommon", package: "PopCommon.SwiftPackage")
				],
            path: "Sources/MsplatCore",
            resources: [
                // Copied, not processed: compiled at runtime unless a precompiled
                // default.metallib is supplied via msplatSetMetallibPath() or found
                // next to the executable / in the bundle.
                .copy("Shaders/msplat_metal.metal"),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
            ]
        ),
        .testTarget(
            name: "MsplatCoreTests",
            dependencies: ["MsplatCore"],
            path: "Tests/MsplatCoreTests"
        ),
    ],
    // Global mutable GPU state (context, tensor cache) mirrors the C++ design and is
    // single-threaded by contract, so build in Swift 5 mode rather than fight strict
    // concurrency checking over it.
    swiftLanguageModes: [.v5]
)
