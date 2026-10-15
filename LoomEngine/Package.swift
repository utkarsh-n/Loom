// swift-tools-version: 6.0
// LoomEngine: the language model itself, UI-free. Foundation + Accelerate only, so it builds and
// tests on the Mac (`Tools/check.sh`) as well as in the iOS app.
import PackageDescription

let package = Package(
  name: "LoomEngine",
  platforms: [.iOS("27.0"), .macOS("15.0")],
  products: [.library(name: "LoomEngine", targets: ["LoomEngine"])],
  targets: [
    .target(
      name: "LoomEngine",
      swiftSettings: [
        .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"]),
        // The engine is arithmetic; unoptimised it is ~20× slower, so Debug app builds optimise it too.
        .unsafeFlags(["-O"], .when(configuration: .debug)),
      ]),
    .testTarget(
      name: "LoomEngineTests", dependencies: ["LoomEngine"],
      resources: [.copy("Fixtures")]),
  ]
)
