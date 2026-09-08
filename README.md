# MixMA-LMM: MixedFrame Memory-Augmented Large Multimodal Model for Long-Term Video Understanding

![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-4169E1)

> [!IMPORTANT]
> **Project status — Research Prototype.**
> MixMA-LMM is an experimental implementation developed for the Digital Breakthrough hackathon. The repository contains the proposed frame-mixing approach, its integration with MA-LMM, demonstration notebooks, and experiment artifacts. It is published as research code and should be validated on the target dataset before production use.

## About the project

This project is a fork of [MA-LMM](https://github.com/boheumd/MA-LMM) that introduces several changes and modifications.

MA-LMM is a model designed for multimodal data and long-term video understanding tasks. It uses a memory mechanism to retain information from earlier parts of a video and apply it to the current task.

## About the modification

One of the main challenges in multimodal video understanding is the large number of frames that must be extracted from a video.

Frame indices are commonly sampled at uniform intervals using `np.linspace`. For long videos, this approach may produce an unrepresentative sample and lose some of the source video's semantics.

ActionShot is a technique for capturing an object in motion and displaying several consecutive appearances of that object in a single image. By combining multiple frames into one, it can preserve semantic information about actions such as movement.

![Example of an ActionShot image](./figs/actionshot_example.jpg)

MixMA-LMM modifies MA-LMM by using a baseline ActionShot approach: multiple frames are blended to imitate a long-exposure image with `cv2.addWeighted`. The algorithm considers both a threshold-based difference between frames and the maximum duration of an action within the camera's field of view.

This repository was created for the Digital Breakthrough hackathon. Further work may explore the effectiveness of this approach.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/xLagerFeuer/mixMA-LMM.git
   ```

2. Enter the project directory:

   ```bash
   cd mixMA-LMM
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Using MA-LMM

1. Train the model or obtain a checkpoint by running the training script for the selected dataset:

   ```bash
   bash run_scripts/${dataset}/train.sh
   ```

2. Evaluate the model:

   ```bash
   bash run_scripts/${dataset}/test.sh ${checkpoint_path}
   ```

3. Use the model for other supported tasks as needed.

## Using MixMA-LMM

- See `main.py` for a basic pipeline example.
- See `mixture/mixMALMM.py`, `mixture/demo_mixmalmm.ipynb`, and `demo_malmm.ipynb` for implementation details and demonstrations.

## References

- [Original MA-LMM repository](https://github.com/boheumd/MA-LMM)
- [MA-LMM paper](https://arxiv.org/abs/2404.05726)

## License

This project is licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause). See the `LICENSE` file in the repository root for the full license text.
