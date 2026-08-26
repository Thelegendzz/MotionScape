# MotionScape

> A Motion-Stratified UAV Video Benchmark for World Modeling and Future Video Generation

[[Hugging Face]](https://huggingface.co/datasets/thelegendzz/MotionScape)
[[Zenodo]](https://doi.org/10.5281/zenodo.21954342)

---

## Overview

MotionScape is a real-world UAV-view video benchmark designed for evaluating future visual continuation and visual prediction under highly dynamic aerial viewpoints. It contains diverse scenes and weather conditions, and is intended to support research on UAV visual prediction, future visual continuation, world model evaluation, spatiotemporal modeling, and text-conditioned video generation.

MotionScape provides clip-level captions that describe scene content, environmental conditions, and camera-motion tendencies. These captions serve as descriptive textual conditions for analyzing future visual generation under aerial motion, and are not intended to serve as explicit action commands or control signals.

## Data Access

The MotionScape benchmark is available through both Hugging Face and Zenodo.

- **Hugging Face:** [MotionScape Dataset](https://huggingface.co/datasets/thelegendzz/MotionScape)
- **Zenodo:** [MotionScape Dataset Record](https://doi.org/10.5281/zenodo.21954342)

The released benchmark contains source video metadata, clip-level semantic annotations, and motion-stratification assignments for 228 benchmark samples. The original audiovisual content is not redistributed. Benchmark samples can be reconstructed from the original publicly accessible source videos using the released metadata and reconstruction code.

Each reconstructed benchmark sample contains 275 consecutive frames standardized to 29.97 FPS, consisting of a 200-frame observed prefix followed by a continuous 75-frame future target.

## License

The released MotionScape metadata, annotations, motion-stratification assignments, and accompanying documentation are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

The original audiovisual content referenced by MotionScape remains subject to the terms and rights of the corresponding source platforms and content owners.

## Citation

If you use MotionScape in your research, please cite the accompanying paper and dataset record.

Dataset DOI: https://doi.org/10.5281/zenodo.21954342

The citation information for the accompanying paper will be updated after publication.
