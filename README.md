# Beyond Static Bias: A Case for Dynamic, Per-Neuron Adaptation in Deep Networks

This repository contains the official PyTorch implementation for the paper "Beyond Static Bias: A Case for Dynamic, Per-Neuron Adaptation in Deep Networks" by Zrng Mahdi Tahir.

## Abstract
The process of feature learning in deep networks often appears random and un-interpretable. To introduce a more structured approach, we propose a method to achieve "Less Randomness" in neural computation through Adaptive Bias Networks (ABN). ABNs replace the monolithic, static bias of a layer with dynamic, per-neuron modulation mechanisms. Each neuron learns to "reason" about the input by generating a context-specific signal that adaptively re-weights its own effective parameters. This is achieved via a gating mechanism that consults a bank of specialized bias vectors. When integrated into a standard ResNet on the CIFAR-10 benchmark, our ABN model demonstrates a consistent performance improvement over the baseline. This suggests that empowering individual neurons with adaptive reasoning capabilities is a more efficient and powerful way to structure model parameters, leading to less random and more effective learning.

## Core Idea: Adaptive Bias Networks (ABN)
Our work introduces the `DynamicModulatedConv2d` layer, which integrates the ABN principle into a standard ResNet. Instead of using a static set of filters, our layer generates an input-dependent modulation signal that dynamically re-weights the output feature maps. This allows the network to adapt its feature extractors for each specific image, leading to enhanced reasoning capabilities.

## Results on CIFAR-10
Our main result shows a consistent improvement over a standard ResNet baseline with a comparable parameter count.

| Model                 | Best Accuracy | Parameters  |
|:----------------------|:--------------|:------------|
| ResNet-S (Standard)   | 91.31%        | 2,777,674   |
| **ResNet-DMB (Ours)** | **91.68%**    | 2,957,794   |

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Adaptive-Bias-Networks.git
    cd Adaptive-Bias-Networks
    ```
    *(Replace YOUR_USERNAME with your actual GitHub username)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the benchmark:**
    ```bash
    python main.py
    ```

## Citation
If you find this work useful in your research, please consider citing our work. The official Zenodo DOI will be linked here upon publication.
