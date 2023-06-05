# Crisis_Vision

Reproduce the results of "Crisis Vision Benchmark Dataset" by Firoj Alam, Ferda Ofli, Muhammad Imran, Tanvirul Alam, Umair Qazi

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
