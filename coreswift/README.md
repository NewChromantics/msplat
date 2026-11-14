# MsplatCore (Swift)

Pure-Swift port of `core/src` (C++ / Objective-C++). The Metal shader
(`msplat_metal.metal`) is unchanged; everything that runs on the CPU — buffer
management, kernel dispatch, the model/optimizer, densification scheduling, dataset
loaders, image I/O, PLY/splat export, checkpoints — is now Swift.

No C++, no nlohmann/json, no nanoflann, no `-lc++`.

## Layout

| Swift file | Replaces |
|---|---|
| `MetalTensor.swift` | `include/metal_tensor.hpp` |
| `MetalContext.swift` | context, metallib loading, command buffers, capture/profiling in `metal/msplat_metal.mm` |
| `MetalPipeline.swift` | `msplat_render`, `msplat_train_step`, `msplat_densify` in `metal/msplat_metal.mm` |
| `Model.swift` | `model.hpp` / `model.cpp` |
| `InputData.swift` | `input_data.hpp` / `input_data.cpp` |
| `Loaders.swift` | `load_colmap.cpp`, `load_nerfstudio.cpp`, `load_polycam.cpp`, `load_ply.cpp` |
| `GaussianIO.swift` | `save_gaussians.cpp` |
| `ImageIO.swift` | `image_io.cpp` |
| `SSIM.swift` | `ssim.cpp` (+ psnr / l1 from `model.cpp`) |
| `KDTree.swift` | `kdtree_tensor.hpp` + `nanoflann.hpp` |
| `Random.swift` | libc++ `mt19937`, `uniform_real_distribution<float>`, `normal_distribution<float>`, `std::shuffle` |
| `API.swift` | `msplat_api.hpp` / `msplat_api.mm` |
| `CAPICompat.swift` | `c_api/msplat_c_api.h` (source-compatible shim) |

## Usage

```swift
import MsplatCore

let ds = try SplatDataset(path: "/data/garden", downscaleFactor: 4, evalMode: true, testEvery: 8)
var cfg = MsplatConfig()
cfg.iterations = 7_000
let trainer = try SplatTrainer(dataset: ds, config: cfg)

trainer.train(callbackEvery: 500) { s in print(s.iteration, s.splatCount, s.msPerStep) }
print(trainer.evaluate())
try trainer.exportPly(to: "garden.ply")
```

### Drop-in for the existing `swift/` wrapper

`CAPICompat.swift` provides the C API names (`msplat_dataset_create`,
`msplat_trainer_step`, `MsplatConfig`, `MsplatPixelBuffer`, …) with the same
argument types, so `swift/Sources/Msplat` builds unchanged. In `swift/Package.swift`
change `.package(path: "../core")` to `.package(path: "../core-swift")` and drop the
`-lc++` linker flag.

## Shader loading

In order: path set via `MsplatRuntime.setMetallibPath(_:)` (or
`msplat_set_metallib_path`) → `default.metallib` in this package's bundle →
`default.metallib` next to the executable → compile the bundled
`Shaders/msplat_metal.metal` at runtime (fast-math, matching `xcrun metal`
defaults). The last step costs a few seconds once per process; ship a precompiled
metallib to avoid it.

`Shaders/msplat_metal.metal` is a copy of `core/src/metal/msplat_metal.metal`; keep
them in sync (or replace the copy with a symlink).

## Fidelity to the C++

* Kernel bindings (buffer indices, scalar widths, `setBytes` layouts, dispatch sizes,
  barriers, blit-zeroing, chunked-raster K_max, the stage-profiling path) are
  one-to-one with `msplat_metal.mm`.
* Random streams are bit-identical to libc++: initial quaternions (`mt19937(42)`),
  camera shuffle order, and the split noise (`mt19937(step)` + normal). The tests
  hold libc++ reference values. Adam bias correction reproduces C++'s
  `std::pow(float, int)` → double promotion.
* KNN scale init is exact, so it matches nanoflann's results.

### Intentional differences

* `MTensor.view()` retains its buffer (the C++ view is non-owning and can dangle).
* `SplatTrainer` retains its dataset (C++ requires the caller to keep it alive).
* Errors are Swift `throws`; the C-compat shim turns them into `fatalError` (the C
  API let C++ exceptions escape, which aborts anyway).
* float→uint8 conversions clamp (C casts were UB on out-of-range/NaN; Swift traps).
* PLY reader only collects properties of the `vertex` element (C++ also counted
  properties of later elements, corrupting the stride if any existed).
* `loadGaussianPly` also recognises the `comment msplat v<step>` line that
  `saveGaussianPly` actually writes (C++ only looked for a different prefix, so the
  step was always 0).
* Nerfstudio `./relative` frame paths fall back to the project root when they don't
  resolve against the working directory (C++ only tried the working directory).
* `imwriteRGB` writes via an RGBX bitmap (CoreGraphics rejects the 24-bpp context the
  C++ asked for). It isn't used by the training path.
* Undistortion only computes the cropped ROI instead of the full frame then cropping
  (same pixels).
* The unused `compute_loss` forward path (`ssim_v_fwd`) is not ported.

## Tests

```
swift test                                     # RNG parity, KNN, SSIM, resize
MSPLAT_TEST_DATASET=/path/to/colmap swift test # + 600-step training smoke test
```
