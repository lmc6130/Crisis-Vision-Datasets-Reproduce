import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb
from torch.nn.parallel import DataParallel


def modify_model_head(model, num_classes):
    model_name = model.__class__.__name__.lower()

    if 'resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'efficientnet' in model_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'densenet' in model_name:
        model.classifier[-1] = nn.Linear(model.classifier.in_features, num_classes)
    elif 'mobilenet' in model_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'vgg' in model_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture for modifying head.")
    
    return model

def create_model(model_type, weights, num_classes, device, freeze=False):
    model_fn_dict = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'effnet': models.efficientnet_b1,
        'mobilenet': models.mobilenet_v2,
        'densenet': models.densenet121,
        'vgg': models.vgg16,
    }

    if model_type not in model_fn_dict:
        raise ValueError("Invalid model type.")

    backbone = model_fn_dict[model_type](weights=weights)
    wandb.log({"checkpoint_info": f"Using {model_type.capitalize()} Model, Freeze Backbone: {freeze}"})
    print("checkpoint:", f"Using {model_type.capitalize()} Model, Freeze Backbone: {freeze}")

    if freeze:
        for name, param in backbone.named_parameters():
            if not any(layer_name in name for layer_name in ('fc', 'classifier')):
                param.requires_grad = False
        print('checkpoint: Backbone Frozen')
        wandb.log({"checkpoint_info": "Backbone Frozen"})
    else:
        print('checkpoint: Fully fine-tuning')
        wandb.log({"checkpoint_info": "Fully Fine-tuning Backbone"})

    model = modify_model_head(backbone, num_classes)

    model = DataParallel(model)
    model = model.to(device)

    return model

def get_model_optimizer_scheduler(args, class_names, device):
    model_weights_dict = {
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
        'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
        'vgg': models.VGG16_Weights.IMAGENET1K_V1,
        'densenet': models.DenseNet121_Weights.IMAGENET1K_V1,
        'mobilenet': models.MobileNet_V2_Weights.IMAGENET1K_V1,
        'effnet': models.EfficientNet_B1_Weights.IMAGENET1K_V1,
    }

    if args.model not in model_weights_dict:
        raise ValueError("Invalid model choice.")

    weights = model_weights_dict[args.model]
    model = create_model(args.model, weights, len(class_names), device, freeze=args.freeze)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='max')

    return model, criterion, optimizer, lr_scheduler