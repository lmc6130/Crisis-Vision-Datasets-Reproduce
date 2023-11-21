# Reproducing Paper: Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response

This repository contains code and resources to reproduce the results of the paper "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response" published in the 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM).

Paper link: https://arxiv.org/abs/2011.08916

## Experiment Setup
* batch_size=128
* epoch=150
* Initial learning rate=0.00001
* optimizer=Adam
* Resolution:
  * DenseNet, MobileNet, ResNet, VGG = 224*224
  * EfficientNet-b1 = 240*240
* GPU : NVIDIA Tesla V100 SXM2 32GB

## Reproduce Results
This experiment uses the F1-score as the metric.
### Damage Severity
| Model | Author Result | Reproduce Results | Reproduce Weight |
| :--- | :---: | :---: | :---: |
| DenseNet-121 | 73.90% | 74.37% | [Weight](https://drive.google.com/file/d/1KTFAi43RMVnwNZ-YRt-kV9WUbGZmovHt/view?usp=drive_link) |
| EfficientNet-b1 | **75.80%** | **76.28%** | [Weight](https://drive.google.com/file/d/1FZUrzAR_qiPYMt5B5Rmn-zGkdazh1NbG/view?usp=drive_link) |
| MobileNet-v2 | 73.00% | 74.21% | [Weight](https://drive.google.com/file/d/1x2BEUhjO5f4IylHCC2sR6SVZa0FV75Yx/view?usp=drive_link) |
| ResNet-18 | 73.60% | 74.04% | [Weight](https://drive.google.com/file/d/13fSwKpZvyDiZOuBpQz1eB9gxdNu6kDzG/view?usp=drive_link) |
| ResNet-50 | 75.10% | 74.03% | [Weight](https://drive.google.com/file/d/1TR8E_EBRVOGmkmzH6YxBqE-Q4uyhFV4j/view?usp=drive_link) |
| ResNet-101 | 73.70% | 73.92% | [Weight](https://drive.google.com/file/d/1nYJngvlDLLuQBhoFueus3k0wzZ3kRGni/view?usp=drive_link) |
| VGG-16 | 75.30% | 75.85% | [Weight](https://drive.google.com/file/d/1aNo4oNys8BPh4GS-3VufiVkuLIrHJp2k/view?usp=drive_link) |

### Disaster Types
| Model | Author Result | Reproduce Results | Reproduce Weight |
| :--- | :---: | :---: | :---: |
| DenseNet-121 | 80.60% | 79.92% | [Weight](https://drive.google.com/file/d/1hQ-uFzD3I6ygkmv_iq4kAGy4g_64Lhph/view?usp=drive_link) |
| EfficientNet-b1 | **81.60%** | **81.48%** | [Weight](https://drive.google.com/file/d/1dmpYsNetbvFQU49em84iG8DIab4WZT5R/view?usp=drive_link) |
| MobileNet-v2 | 78.20% | 78.59% | [Weight](https://drive.google.com/file/d/1AUt0RI78Encoo7rYEoo8QBQaqyK8KwM_/view?usp=drive_link) |
| ResNet-18 | 78.50% | 78.59% | [Weight](https://drive.google.com/file/d/1_ykONq7K0djuSq0W_JKOE3H-_2IxSm7d/view?usp=drive_link) |
| ResNet-50 | 80.80% | 80.01% | [Weight](https://drive.google.com/file/d/1K4STGU7KZ9XFCdEqMtK2mgtjzIBbc7K8/view?usp=drive_link) |
| ResNet-101 | 81.30% | 80.36% | [Weight](https://drive.google.com/file/d/1DP7Wd1J2J7qAhb3acUo6gdZxwnv7i-GE/view?usp=drive_link) |
| VGG-16 | 79.80% | 80.04% | [Weight](https://drive.google.com/file/d/1KVDijAFzUaX9KJm1lrcpIyRy9g2L6xap/view?usp=drive_link) |

### Humanitarian
| Model | Author Result | Reproduce Results | Reproduce Weight |
| :--- | :---: | :---: | :---: |
| DenseNet-121 | 75.50% |  |  |
| EfficientNet-b1 | 76.50% |  |  |
| MobileNet-v2 | 74.60% |  |  |
| ResNet-18 | 74.90% |  |  |
| ResNet-50 | 76.20% |  |  |
| ResNet-101 | 76.50% |  |  |
| VGG-16 | **77.30%** |  |  |

### Informative
| Model | Author Result | Reproduce Results | Reproduce Weight |
| :--- | :---: | :---: | :---: |
| DenseNet-121 | 86.20% |  |  |
| EfficientNet-b1 | **86.30%** |  |  |
| MobileNet-v2 | 84.90% |  |  |
| ResNet-18 | 85.10% |  |  |
| ResNet-50 | 85.20% |  |  |
| ResNet-101 | 85.20% |  |  |
| VGG-16 | 85.80% |  |  |

## Implementated Network
* DenseNet      [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
* EfficientNet  [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* MobileNet-v2  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* ResNet        [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
* VGG           [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)

## Dataset
* [Crisis Vision Dataset](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20)

## Citation
If you find this code or dataset useful in your research, please consider citing the original paper:

F. Alam, F. Ofli, M. Imran, T. Alam and U. Qazi, "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response," 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), The Hague, Netherlands, 2020, pp. 151-158, doi: 10.1109/ASONAM49781.2020.9381294.

Please note that this code is provided for research purposes only and should be used responsibly.
