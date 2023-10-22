# Reproducing Paper: Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response

This repository contains code and resources to reproduce the results of the paper "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response" published in the 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM).

Paper link: https://arxiv.org/abs/2011.08916

## Experiment Setup
* batch_size=128
* epoch=150
* Initial learning rate=0.00001
* optimizer=Adam
* Resolution:
  * DenseNet = 224*224
  * EfficientNet-b1 = 240*240
  * ResNet = 224*224
* GPU : NVIDIA Tesla V100 SXM2 32GB

## Reproduce Results
This experiment uses the F1-score as the metric.
### Damage Severity
| Model | Author Result | Reproduce Results
| :--- | :---: | :---: |
| DenseNet-121 | 73.90% |  |
| EfficientNet-b1 | **75.80%** | **75.89%** |
| ResNet-18 | 73.60% |  |
| ResNet-50 | 75.10% | 74.69% |
| ResNet-101 | 73.70% |  |
| VGG-16 | 75.30% |  |

### Disaster Types
| Model | Author Result | Reproduce Results
| :--- | :---: | :---: |
| DenseNet-121 | 80.60% |  |
| EfficientNet-b1 | **81.60%** |  |
| ResNet-18 | 78.50% |  |
| ResNet-50 | 80.80% |  |
| ResNet-101 | 81.30% |  |
| VGG-16 | 79.80% |  |

### Humanitarian
| Model | Author Result | Reproduce Results
| :--- | :---: | :---: |
| DenseNet-121 | 75.50% |  |
| EfficientNet-b1 | 76.50% |  |
| ResNet-18 | 74.90% |  |
| ResNet-50 | 76.20% |  |
| ResNet-101 | 76.50% |  |
| VGG-16 | **77.30%** |  |

### Informative
| Model | Author Result | Reproduce Results
| :--- | :---: | :---: |
| DenseNet-121 | 86.20% |  |
| EfficientNet-b1 | **86.30%** |  |
| ResNet-18 | 85.10% |  |
| ResNet-50 | 85.20% |  |
| ResNet-101 | 85.20% |  |
| VGG-16 | 85.80% |  |

## Implementated Network
* DenseNet      [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
* EfficientNet  [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* ResNet        [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
* VGG           [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)

## Dataset
* [Crisis Vision Dataset](https://crisisnlp.qcri.org/crisis-image-datasets-asonam20)

## Citation
If you find this code or dataset useful in your research, please consider citing the original paper:

F. Alam, F. Ofli, M. Imran, T. Alam and U. Qazi, "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response," 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), The Hague, Netherlands, 2020, pp. 151-158, doi: 10.1109/ASONAM49781.2020.9381294.

Please note that this code is provided for research purposes only and should be used responsibly.
