import XCTest
@testable import MsplatCore

final class RandomParityTests: XCTestCase {
    // Reference values produced by the libc++ implementations used by the C++ core.

    func testMT19937() {
        var g = MT19937(seed: 42)
        XCTAssertEqual((0..<5).map { _ in g.next() },
                       [1608637542, 3421126067, 4083286876, 787846414, 3143890026])
        var d = MT19937()
        for _ in 0..<9999 { _ = d.next() }
        XCTAssertEqual(d.next(), 4123659995)
    }

    func testUniformRealFloat() {
        var g = MT19937(seed: 42)
        let u = UniformRealFloat(0, 1)
        XCTAssertEqual((0..<6).map { _ in u.sample(&g).bitPattern },
                       [0x3ebfc3b9, 0x3f4bea3e, 0x3f736203, 0x3e3bd654, 0x3f3b63f4, 0x3f4799d4])
    }

    func testNormalFloat() {
        var g = MT19937(seed: 600)
        var n = NormalFloat()
        XCTAssertEqual((0..<7).map { _ in n.sample(&g).bitPattern },
                       [0x3f42cc9c, 0x400c8ee1, 0x3f39ba2d, 0x3f1e8a1a, 0xbfc0d861, 0x3fbca562, 0xbfb6e30d])
    }

    func testShuffle() {
        func twice(_ n: Int) -> [Int] {
            var a = Array(0..<n)
            var g = MT19937(seed: 42)
            shuffleLibcxx(&a, &g); shuffleLibcxx(&a, &g)
            return a
        }
        XCTAssertEqual(twice(10), [0, 5, 2, 3, 7, 6, 8, 1, 4, 9])
        XCTAssertEqual(Array(twice(200).prefix(12)), [111, 62, 153, 135, 124, 69, 191, 97, 183, 150, 16, 18])
        XCTAssertEqual(twice(1), [0])
    }
}

final class CPUTests: XCTestCase {
    func testKnnScalesMatchesBruteForce() {
        var g = MT19937(seed: 7)
        let u = UniformRealFloat(-1, 1)
        let n = 2000
        var xyz = (0..<(n * 3)).map { _ in u.sample(&g) }
        for i in 100..<150 { for c in 0..<3 { xyz[i * 3 + c] = xyz[c] } }  // duplicates
        let fast = knnScales(xyz: xyz, count: n)
        for i in stride(from: 0, to: n, by: 37) {
            var d: [Float] = []
            for j in 0..<n {
                var r: Float = 0
                for c in 0..<3 { let t = xyz[i * 3 + c] - xyz[j * 3 + c]; r += t * t }
                d.append(r)
            }
            d.sort()
            let ref = (d[1].squareRoot() + d[2].squareRoot() + d[3].squareRoot()) / 3
            XCTAssertEqual(fast[i], ref, accuracy: 1e-6, "point \(i)")
        }
    }

    func testResizeAreaHalvesExactly() {
        var img = RGBImage(data: [], width: 4, height: 2)
        img.data = (0..<(4 * 2 * 3)).map { Float($0) }
        let out = resizeArea(img, 2, 1)
        XCTAssertEqual(out.width, 2); XCTAssertEqual(out.height, 1)
        // Mean of the 2x2 block for pixel 0, channel 0: (0 + 3 + 12 + 15) / 4
        XCTAssertEqual(out.data[0], 7.5, accuracy: 1e-5)
    }

    func testSSIMOfIdenticalImagesIsOne() {
        let t = MTensor(cpuShape: [16, 16, 3], dtype: .float32)
        let p = t.floats()
        for i in 0..<t.numel { p[i] = Float(i % 7) / 7 }
        XCTAssertEqual(ssimEval(t, t), 1, accuracy: 1e-5)
        XCTAssertEqual(l1Loss(t, t), 0)
    }

    func testSSIMWindowSumsToOne() {
        XCTAssertEqual(createSSIMWindow().reduce(0, +), 1, accuracy: 1e-5)
    }
}

/// End-to-end smoke test on a real dataset (needs a Metal GPU).
/// Set MSPLAT_TEST_DATASET to a COLMAP / Nerfstudio / Polycam folder to enable.
final class TrainingSmokeTests: XCTestCase {
    func testTrainFewSteps() throws {
        guard let path = ProcessInfo.processInfo.environment["MSPLAT_TEST_DATASET"] else {
            throw XCTSkip("MSPLAT_TEST_DATASET not set")
        }
        let ds = try SplatDataset(path: path, downscaleFactor: 4, evalMode: true, testEvery: 8)
        var cfg = MsplatConfig()
        cfg.iterations = 600
        let trainer = try SplatTrainer(dataset: ds, config: cfg)
        let start = trainer.splatCount
        trainer.train()
        XCTAssertEqual(trainer.iteration, 600)
        XCTAssertGreaterThan(trainer.splatCount, 0)
        print("splats: \(start) -> \(trainer.splatCount)")

        let m = trainer.evaluate()
        XCTAssertTrue(m.psnr.isFinite && m.psnr > 10, "psnr \(m.psnr)")

        let img = try XCTUnwrap(trainer.render(cameraIndex: 0))
        XCTAssertEqual(img.pixels.count, img.width * img.height * 3)

        let dir = FileManager.default.temporaryDirectory
        let ckpt = dir.appendingPathComponent("msplat-test.ckpt").path
        try trainer.saveCheckpoint(to: ckpt)
        XCTAssertEqual(try trainer.loadCheckpoint(from: ckpt), 600)
        try trainer.exportPly(to: dir.appendingPathComponent("msplat-test.ply").path)
        try trainer.exportSplat(to: dir.appendingPathComponent("msplat-test.splat").path)
        MsplatRuntime.cleanup()
    }
}
