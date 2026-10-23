// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "core",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MsplatCore", targets: ["MsplatCore"]),
    ],
    targets: [
        .target(
            name: "MsplatCore",
            path: "src",
            resources: [
				//.copy("Resources/default.metallib")
			],
            cxxSettings: [
				.headerSearchPath("include"),
				.headerSearchPath("metal"),
                .unsafeFlags(["-std=c++17"]),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Foundation"),
                .linkedFramework("ImageIO"),
                .linkedFramework("CoreGraphics"),
                .unsafeFlags(["-lc++"]),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
