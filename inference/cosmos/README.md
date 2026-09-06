# Cosmos-Predict2.5 integration

This directory records the MotionScape integration for the upstream
Cosmos-Predict2.5 repository:

- `BASE_COMMIT` contains the exact upstream Git commit used by the benchmark.
- `motionscape.patch` contains the MotionScape inference adaptations applied to
  that clean upstream revision.
- `assets/` contains the task configurations and their referenced prompt files
  for Text2World, Image2World, and Video2World inference.

Apply the patch from a clean upstream checkout:

```bash
cd /path/to/cosmos-predict2.5
git checkout "$(cat /path/to/MotionScape/inference/cosmos/BASE_COMMIT)"
git apply --check /path/to/MotionScape/inference/cosmos/motionscape.patch
git apply /path/to/MotionScape/inference/cosmos/motionscape.patch
```

The JSON files under `assets/` use prompt paths relative to that directory.
Keep each JSON file and its referenced prompt together when passing the assets
to the patched upstream `examples/inference.py` entry point.
