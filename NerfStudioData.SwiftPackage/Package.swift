// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NerfStudioData",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "NerfStudioData", targets: ["NerfStudioData"]),
    ],
    targets: [
        .target(
            name: "NerfStudioData",
            path: "."
        )
    ],
)
