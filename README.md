# Reproducing Paper: Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response

This repository contains code and resources to reproduce the results of the paper "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response" published in the 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM).

Paper link: https://arxiv.org/abs/2011.08916

## Reproduce Experiment Setup
* batch_size=128
* epoch=100
* optimizer=Adam
* Initial learning rate=0.00001
* lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
* GPU : NVIDIA Tesla V100 SXM2 32GB
* Model: ResNet-18, ResNet-50, ResNet-101, VGG-16, DenseNet-121, EfficientNet-b1

## Implementated Network
* ResNet        [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
* EfficientNet  [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* VGG           [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
* DenseNet      [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)

## Citation
If you find this code or dataset useful in your research, please consider citing the original paper:

F. Alam, F. Ofli, M. Imran, T. Alam and U. Qazi, "Deep Learning Benchmarks and Datasets for Social Media Image Classification for Disaster Response," 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), The Hague, Netherlands, 2020, pp. 151-158, doi: 10.1109/ASONAM49781.2020.9381294.

Please note that this code is provided for research purposes only and should be used responsibly.
