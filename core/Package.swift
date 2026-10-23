// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "core",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MsplatCore", targets: ["MsplatCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/NewChromantics/nlohmann_json", branch: "graham/FixSwiftPackage")
    ],
    targets: [
        .target(
            name: "MsplatCore",
            dependencies: [
                .product(name: "json", package: "nlohmann_json")
            ],
            path: "src",
            resources: [
				//.copy("Resources/default.metallib")
			],
            cxxSettings: [
				.headerSearchPath("include"),
				.headerSearchPath("metal"),
                .define("_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES", to: "1"),
                .unsafeFlags(["-std=c++17", "-stdlib=libc++"]),
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
