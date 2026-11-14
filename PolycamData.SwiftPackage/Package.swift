// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PolycamData",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "PolycamData", targets: ["PolycamData"]),
    ],
    targets: [
        .target(
            name: "PolycamData",
            path: "."
        )
    ],
)
