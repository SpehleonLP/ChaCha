# ChaCha

Joint articulation constraint analysis from skeletal animation data.

ChaCha analyzes animation keyframes to deduce per-joint constraint descriptors: range of motion, velocity limits, and effort limits for each degree of freedom. Output follows the [AGI_articulations](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/AGI_articulations) glTF extension schema.

## Usage

```cpp
#include <chacha.h>

// Prepare input: animation channels + skeleton hierarchy
std::vector<ChaCha::AnimationChannel> channels = /* extract from your data */;
ChaCha::Skeleton skeleton = /* parents array + rest poses */;

// Analyze
auto articulations = ChaCha::analyze(channels, skeleton);

// Each articulation describes one joint's constraints
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

1. Convert keyframes to rest-pose-relative values
2. Compute per-keyframe velocity derivatives
3. Decompose rotations via swing-twist into independent DOFs (xRotate, yRotate, zRotate)
4. Segment and merge motion ranges across all animations per joint
5. Filter out noise below configurable thresholds
6. Produce one `Articulation` per joint with significant motion

## Configuration

```cpp
ChaCha::Options options;
options.rotation_threshold_rad = 0.01f;   // Min rotation range to keep (radians)
options.translation_threshold_m = 0.001f; // Min translation range to keep (meters)
options.scale_threshold = 0.01f;          // Min scale range to keep
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
