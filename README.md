# Reproduction of Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response

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
| DenseNet-121 | 75.50% | 75.55% | [Weight](https://drive.google.com/file/d/1izWizcQcexbJaNZ9VplCjj_eEiDMr7dR/view?usp=drive_link) |
| EfficientNet-b1 | 76.50% | **77.08%** | [Weight](https://drive.google.com/file/d/1q1fWpdIeTbPYDsnRQb2kgoeFkgDkR759/view?usp=drive_link) |
| MobileNet-v2 | 74.60% | 75.29% | [Weight](https://drive.google.com/file/d/1ik7rd-aRsh220v2G5q0oRfBtoFpYxGXn/view?usp=drive_link) |
| ResNet-18 | 74.90% | 74.62% | [Weight](https://drive.google.com/file/d/19ifr44PnEUIWnn6JGHWxTc-St-SRvylJ/view?usp=drive_link) |
| ResNet-50 | 76.20% | 76.43% | [Weight](https://drive.google.com/file/d/1W3_1c6ZkziqLyYuCf_P9IB7S09kCBEB4/view?usp=drive_link) |
| ResNet-101 | 76.50% | 76.16% | [Weight](https://drive.google.com/file/d/1k3kJRrJRu8kwhopKuo2cZ3fJQ5jHwb5u/view?usp=drive_link) |
| VGG-16 | **77.30%** | 76.01% | [Weight](https://drive.google.com/file/d/1eRKBCkFD2eFm5jnoZPgPlSKGipctizQx/view?usp=drive_link) |

### Informative
| Model | Author Result | Reproduce Results | Reproduce Weight |
| :--- | :---: | :---: | :---: |
| DenseNet-121 | 86.20% | 85.06% | [Weight](https://drive.google.com/file/d/1ehyCmaHwFEcQKYyp0eubtqnN9zfJgMco/view?usp=drive_link) |
| EfficientNet-b1 | **86.30%** | 85.74% | [Weight](https://drive.google.com/file/d/1aH1E7KhY9i_cX_JXcFAxjyHNLiJ1L40I/view?usp=drive_link) |
| MobileNet-v2 | 84.90% | 85.04% | [Weight](https://drive.google.com/file/d/181rpcc70cE7zM4a8rpVq1BULnoEyX2xe/view?usp=drive_link) |
| ResNet-18 | 85.10% | 84.56% | [Weight](https://drive.google.com/file/d/14ygTIGLDzQlD6Z7XIn-f3U5CbsQkx2r7/view?usp=drive_link) |
| ResNet-50 | 85.20% | 85.36% | [Weight](https://drive.google.com/file/d/1DR2vlAlj0ETGjgulPlc_IJ_5eHs92djK/view?usp=drive_link) |
| ResNet-101 | 85.20% | 85.42% | [Weight](https://drive.google.com/file/d/1En6PgZZDc74Kn7eua_MenpVQMfL9erZc/view?usp=drive_link) |
| VGG-16 | 85.80% | **85.80%** | [Weight](https://drive.google.com/file/d/1AfHoTPDEYrUY01PsOj6RS2Yeqm8KHR5L/view?usp=drive_link) |

## Implementated Network
* DenseNet      [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
* EfficientNet  [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* MobileNet-v2  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* ResNet        [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
* VGG           [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)

## Dataset
* [Crisis Vision Dataset](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20)

## Citation
If you use the code and pre-trained weights from this repository in your research, please cite this work as the source of the reproduced weights:

```
@misc{lim2023dsarepro,
  title={Reproduction of Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response},
  author={Lim, Ming Chung},
  year={2023},
  howpublished={GitHub repository},
  url={https://github.com/lmc6130/Disaster-Response-Datasets-Reproduce},
}
```

These pre-trained weights are based on the reproduction of the paper "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response" by F. Alam et al. This code is provided for research purposes only. Please refer to the original paper for a comprehensive understanding of the methods and results.

```
@inproceedings{alam2020deep,
  title={Deep learning benchmarks and datasets for social media image classification for disaster response},
  author={Alam, Firoj and Ofli, Ferda and Imran, Muhammad and Alam, Tanvirul and Qazi, Umair},
  booktitle={2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  pages={151--158},
  year={2020},
  organization={IEEE}
}
```
