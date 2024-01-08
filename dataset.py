import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

def create_datasets(data_dir, resize=(224, 224)):
    tr_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    class_names = os.listdir(os.path.join(data_dir, "train"))

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_train_transform(tr_normalize, resize))
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=get_common_transform(tr_normalize, resize))
    dev_dataset = datasets.ImageFolder(os.path.join(data_dir, "dev"), transform=get_common_transform(tr_normalize, resize))

    return train_dataset, test_dataset, dev_dataset, class_names

def get_common_transform(normalize_transform, resize=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize_transform,
    ])

def get_train_transform(normalize_transform, resize=(224, 224), use_trivial_aug=None):
    transform_list = [
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        # transforms.RandomResizedCrop(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize_transform,
    ]
    
    return transforms.Compose(transform_list)

def load_datasets(args):
    return create_datasets(args.data_dir, resize=(args.image_res, args.image_res))

def get_data_loaders(args):
    train_dataset, test_dataset, dev_dataset, class_names = load_datasets(args)
    print('class name:', class_names)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    devloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return trainloader, testloader, devloader, class_names