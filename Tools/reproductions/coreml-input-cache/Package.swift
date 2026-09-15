// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "CoreMLInputCacheReproduction",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "../../..")],
    targets: [
        .executableTarget(
            name: "Reproduce",
            dependencies: [.product(name: "Inference", package: "desert-ant-core")],
            path: ".",
            exclude: ["README.md", "generate_fixture.py"],
            sources: ["Reproduce.swift"],
            resources: [.copy("Resources/identity.mlmodel")]
        ),
    ]
)
