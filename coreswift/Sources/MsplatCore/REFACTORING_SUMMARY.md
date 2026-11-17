# Refactoring Summary: Removing Globals from MetalPipeline.swift

## Goal
Remove global variables and move `gTensorCache` and related state into `MetalContext`, making functions more pure and testable.

## Changes Completed

### 1. MetalContext.swift
- ✅ Added `FusedTensorCache` struct with all methods moved from MetalPipeline.swift
- ✅ Added `OverflowCheck` struct 
- ✅ Added instance variables to `MetalContext`:
  ```swift
  var tensorCache = FusedTensorCache()
  var renderOverflowCheck = OverflowCheck()
  var trainOverflowCheck = OverflowCheck()
  var renderDiagCount = 0
  var trainDiagIt = 0
  ```
- ✅ Added `cleanupTensorCache()` method to MetalContext

### 2. MetalPipeline.swift - Partial Changes
- ✅ Removed duplicate `FusedTensorCache` struct definition
- ✅ Removed duplicate `OverflowCheck` struct definition  
- ✅ Updated helper function signatures to accept `cache` parameter:
  - `encodeProjSh(..., _ cache: FusedTensorCache)`
  - `encodePrefixMap(..., _ cache: FusedTensorCache)`
  - `encodeRastFwd(..., _ cache: FusedTensorCache)`
- ✅ Updated these functions to use `cache` instead of `gTensorCache`

## Changes Still Needed

### MetalPipeline.swift - Remaining Work

1. **Remove old globals** (lines ~169-174):
   ```swift
   // DELETE THESE:
   nonisolated(unsafe) var gTensorCache = FusedTensorCache()
   func cleanupMsplatMetal() { gTensorCache = FusedTensorCache() }
   ```

2. **Update msplatRender function**: Already partially done, may have some remaining refs

3. **Update msplatTrainStep function** to:
   - Replace `trainOverflow` with `ctx.trainOverflowCheck`
   - Replace `trainDiagIt` with `ctx.trainDiagIt`
   - Replace all `gTensorCache` with `ctx.tensorCache`
   - Update all nested function calls to pass `ctx.tensorCache` as parameter
   - Update nested functions (encodeLossFwdBwd, encodeRastBwd, encodeProjShBwdAdam, encodeGradStats, blitZero) to accept cache parameter

4. **Update msplatDensify function** - appears to not use `gTensorCache` directly

## Detailed Remaining Replacements

### In msplatTrainStep:

Replace beginning:
```swift
func msplatTrainStep(_ p: RenderParams, _ t: TrainParams, _ ctx: MetalContext) async throws -> (radii: MTensor, loss: Float) {
    trainOverflow.run(numPoints: p.numPoints, ctx: ctx)
```

With:
```swift
func msplatTrainStep(_ p: RenderParams, _ t: TrainParams, _ ctx: MetalContext) async throws -> (radii: MTensor, loss: Float) {
    ctx.trainOverflowCheck.run(numPoints: p.numPoints, cache: ctx.tensorCache)
```

Replace diagnostics section:
```swift
trainDiagIt += 1
if [200, 1000, 2500, 4500, 6500].contains(trainDiagIt)
    && gTensorCache.tileScatterCounters.defined && gTensorCache.fwdNumPoints > 0
    && benchmarkEnabled {
    let c = gTensorCache
```

With:
```swift  
ctx.trainDiagIt += 1
if [200, 1000, 2500, 4500, 6500].contains(ctx.trainDiagIt)
    && ctx.tensorCache.tileScatterCounters.defined && ctx.tensorCache.fwdNumPoints > 0
    && benchmarkEnabled {
    let c = ctx.tensorCache
```

Replace cache calls:
```swift
gTensorCache.ensureForward(...)
gTensorCache.ensureBackward(...)
gTensorCache.ensureChunks(...)
let c = gTensorCache
```

With:
```swift
ctx.tensorCache.ensureForward(...)
ctx.tensorCache.ensureBackward(...)
ctx.tensorCache.ensureChunks(...)
let c = ctx.tensorCache
```

### Update nested functions in msplatTrainStep:

All nested functions need to accept cache parameter:
```swift
func encodeLossFwdBwd(_ enc: MTLComputeCommandEncoder, _ cache: FusedTensorCache)
func encodeRastBwd(_ enc: MTLComputeCommandEncoder, _ cache: FusedTensorCache)
func encodeProjShBwdAdam(_ enc: MTLComputeCommandEncoder, _ cache: FusedTensorCache)
func encodeGradStats(_ enc: MTLComputeCommandEncoder, _ cache: FusedTensorCache)
func blitZero(_ cb: MTLCommandBuffer, _ cache: FusedTensorCache)
```

And replace all `c.xxx` references with `cache.xxx` in those functions.

Then call them with:
```swift
blitZero(cb, ctx.tensorCache)
encodeLossFwdBwd(enc, ctx.tensorCache)
encodeRastBwd(enc, ctx.tensorCache)
encodeProjShBwdAdam(enc, ctx.tensorCache)
encodeGradStats(enc, ctx.tensorCache)
```

## Benefits

After completion, this refactoring will:

1. **Eliminate global mutable state** - Making code thread-safe and testable
2. **Make dependencies explicit** - Functions clearly show what state they need  
3. **Enable multiple contexts** - Can have multiple independent Metal contexts
4. **Improve testability** - Can inject mock/test tensor caches
5. **Better encapsulation** - State lives where it belongs (in MetalContext)

## Migration Guide for Callers

### Before:
```swift
cleanupMsplatMetal()  // Global function
let result = try await msplatRender(params, ctx)
```

### After:
```swift
ctx.cleanupTensorCache()  // Method on context
let result = try await msplatRender(params, ctx)  // Same call signature
```

The public API (`msplatRender`, `msplatTrainStep`, `msplatDensify`) remains unchanged - only internal implementation is cleaner.
