// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ColmapData",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "ColmapData", targets: ["ColmapData"]),
    ],
    targets: [
        .target(
            name: "ColmapData",
            path: "."
        )
    ],
)
