# ChaCha

Joint articulation constraint analysis from skeletal animation data.

ChaCha analyzes animation keyframes to deduce per-joint constraint descriptors: range of motion, observed velocity, and observed peak acceleration for each degree of freedom. Output follows the [AGI_articulations](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/AGI_articulations) glTF extension schema.

## Usage

```cpp
#include <chacha.h>

// Prepare input: animation channels, animation names, and skeleton hierarchy
std::vector<ChaCha::AnimationChannel> channels   = /* extract from your data */;
std::vector<ChaCha::Animation>        animations = /* one entry per animation clip */;
ChaCha::Skeleton skeleton = /* parents array + rest poses */;

// Analyze (options, scan, and diagnostics are optional)
auto articulations = ChaCha::analyze(channels, animations, skeleton);

// Each articulation describes one joint's constraints. `stages` is an ORDERED
// SEQUENCE -- a StageType can legitimately repeat (proper-Euler charts), so
// never key or deduplicate stages by type.
for (const auto& art : articulations) {
    printf("Joint %d (%s): %zu DOFs\n",
           art.node, art.name.c_str(), art.stages.size());
    for (const auto& stage : art.stages) {
        printf("  %s: [%.2f, %.2f] max_vel=%.2f\n",
               ChaCha::stage_type_name(stage.type),
               stage.min_value, stage.max_value, stage.max_velocity);
    }
}
```

## How It Works

1. Convert keyframes to rest-pose-relative values (rotation: subtract rest quaternion;
   translation/scale: subtract/divide by rest translation/scale).
2. Decompose rotation per joint: if the scan narrowed to an artist-authored "AGI "
   configuration animation with clean one-axis-at-a-time motion, use its phase order
   directly; otherwise search every 1-axis, 2-axis, and 3-axis (12 Euler chart) decomposition
   and keep the best fit under a residual gate.
3. Combine multiple animations of the same joint by anchoring each against a common
   reference before unioning ranges, so trajectories that straddle a +-pi wrap don't
   collapse to a spurious ~2pi span.
4. Compute velocity and acceleration by resampling onto a uniform time grid.
5. Filter out stages below configurable thresholds.
6. Infer a model-wide pointing vector and produce one `Articulation` per joint with
   significant motion.

## Configuration

```cpp
ChaCha::Options options;
options.rotation_threshold_rad = 0.01f;    // Min rotation range to keep (radians)
options.translation_threshold_m = 0.001f;  // Min translation range to keep (meters)
options.scale_threshold = 0.01f;           // Min scale range to keep
options.prioritize_rom_animations = true;  // Auto-narrow scan to "AGI "-prefixed animations
options.resample_rate_hz = 60.0f;          // Uniform grid used for velocity/acceleration
options.derivative_window = 5;             // Smoothing window for derivative estimates
options.max_fit_residual_rad = 0.02f;      // Acceptance gate for reduced (1-/2-axis) candidates
```

## Building

Standalone:

```bash
mkdir build && cd build
cmake ..
make
```

As a submodule (typical usage):

```cmake
add_subdirectory(modules/chacha)
target_link_libraries(your_target PRIVATE chacha)
```

Requires CMake 3.15+, C++20, and GLM.

## Dependencies

- **GLM** (vec3, quat) — found via `find_package` or assumed in include path
- **C++20** standard library (std::span)
- No other dependencies. No file I/O, no JSON, no GPU.

## Integration

ChaCha operates on pre-extracted animation data and produces structured constraint data. Bridge code to read/write specific file formats (e.g., glTF) lives in the consuming application.

See the [tonton-example](https://github.com/SpehleonLP/tonton-example) project for a complete integration example with glTF via fx-gltf.

## License

Apache-2.0. See [LICENSE](LICENSE).
