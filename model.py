import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb
from torch.nn.parallel import DataParallel


def resnet18_model(weights, num_classes):
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def resnet50_model(weights, num_classes):
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def resnet101_model(weights, num_classes):
    model = models.resnet101(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def efficientnetb1_model(weights, num_classes):
    model = models.efficientnet_b1(weights=weights)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model

def mobilenetv2_model(weights, num_classes):
    model = models.mobilenet_v2(weights=weights)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model

def densenet121_model(weights, num_classes):
    model = models.densenet121(weights=weights)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def vgg16_model(weights, num_classes):
    model = models.vgg16(weights=weights)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model

def create_model(model_type, weights, num_classes, device, freeze=False):
    if model_type == 'resnet18':
        backbone = resnet18_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using ResNet-18 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using ResNet-18 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'fc' in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    elif model_type == 'resnet50':
        backbone = resnet50_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using ResNet-50 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using ResNet-50 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'fc' in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    elif model_type == 'resnet101':
        backbone = resnet101_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using ResNet-101 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using ResNet-101 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'fc' in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    elif model_type == 'effnet':
        backbone = efficientnetb1_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using EfficientNet-b1 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using EfficientNet-b1 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'classifier' in name:
                    param.requires_grad = False
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})
    
    elif model_type == 'mobilenet':
        backbone = mobilenetv2_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using MobileNet-v2 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using MobileNet-v2 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'classifier' in name:
                    param.requires_grad = False
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    elif model_type == 'vgg':
        backbone = vgg16_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using VGG-16 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using VGG-16 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'classifier' in name:
                    param.requires_grad = False
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    elif model_type == 'densenet':
        backbone = densenet121_model(weights, num_classes)
        wandb.log({"checkpoint_info": f"Using DenseNet-121 Model, Freeze Backbone: {freeze}"})
        print("checkpoint:", f"Using DenseNet-121 Model, Freeze Backbone: {freeze}")
        if freeze:
            for name, param in backbone.named_parameters():
                if not 'classifier' in name:
                    param.requires_grad = False
            print('checkpoint: Backbone Frozen')
            wandb.log({"checkpoint_info": "Backbone Frozen"})
        else:
            print('checkpoint: fully fine-tuning')
            wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    else:
        raise ValueError("Invalid model type.")

    model = DataParallel(model)
    model = model.to(device)

    return model

def get_model_optimizer_scheduler(args, class_names, device):
    if args.model == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    elif args.model == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    elif args.model == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    elif args.model == 'vgg':
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    elif args.model == 'densenet':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    elif args.model == 'mobilenet':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    elif args.model == 'effnet':
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
        model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')
    else:
        raise ValueError("Invalid model choice.")

    return model, criterion, optimizer, lr_scheduler