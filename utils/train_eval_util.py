import os
import torch
import torchvision
from transformers import CLIPModel, AlignModel, GroupViTModel, AltCLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011, \
            Cub100, OxfordIIITPet_18, Food101_50, StanfordCars98
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def set_model_clip(args):
    if args.model == 'CLIP':
        '''
        load Huggingface CLIP
        '''
        ckpt_mapping = {"ViT-B/16":"openai/clip-vit-base-patch16", 
                        "ViT-B/32":"openai/clip-vit-base-patch32",
                        "ViT-L/14":"openai/clip-vit-large-patch14"}
        args.ckpt = ckpt_mapping[args.CLIP_ckpt]
        model =  CLIPModel.from_pretrained(args.ckpt)
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        val_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
    elif args.model == 'ALIGN':
        model = AlignModel.from_pretrained("kakaobrain/align-base")
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))  # for ALIGN
        val_preprocess = transforms.Compose([
                transforms.Resize(289),
                transforms.CenterCrop(289),
                transforms.ToTensor(),
                normalize
            ])
    elif args.model == 'GroupViT':
        model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)  # for GroupViT
        val_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
    elif args.model == 'AltCLIP':
        model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        val_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

    model = model.cuda()
    return model, val_preprocess


def set_val_loader(args, preprocess=None):
    def create_loader(dataset, root, preprocess, batch_size, shuffle, **kwargs):
        return torch.utils.data.DataLoader(
            dataset(root, transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs
        )
    
    root = args.root_dir
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    kwargs = {'num_workers': 4, 'pin_memory': True}
    dataset_paths = {
        "ImageNet": os.path.join(root, 'ImageNet', 'val'),
    }
    
    dataset_classes = {
        "ImageNet10": lambda root, transform: datasets.ImageFolder(os.path.join(root, "ImageNet10", 'val'), transform=transform),
        "ImageNet20": lambda root, transform: datasets.ImageFolder(os.path.join(root, "ImageNet20", 'val'), transform=transform),
        "bird200": lambda root, transform: Cub2011(root, train=False, transform=transform),
        "car196": lambda root, transform: StanfordCars(root, split="test", download=True, transform=transform),
        "food101": lambda root, transform: Food101(root, split="test", download=True, transform=transform),
        "pet37": lambda root, transform: OxfordIIITPet(root, split="test", download=True, transform=transform),
        "cub100_ID": lambda root, transform: Cub100(root, train=False, id=True, transform=transform),
        "car98_ID": lambda root, transform: StanfordCars98(root, split="test", id=True, download=True, transform=transform),
        "food50_ID": lambda root, transform: Food101_50(root, split="test", id=True, download=True, transform=transform),
        "pet18_ID": lambda root, transform: OxfordIIITPet_18(root, id=True, split="test", download=True, transform=transform),
        "cifar10": lambda root, transform: datasets.CIFAR10(root=os.path.join(root, "cifar10"), train=False, download=True, transform=transform),
        "cifar100": lambda root, transform: datasets.CIFAR100(root=os.path.join(root, "cifar100"), train=False, download=True, transform=transform)
    }
    
    if args.in_dataset in dataset_paths:
        path = dataset_paths[args.in_dataset]
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    elif args.in_dataset in dataset_classes:
        val_loader = create_loader(dataset_classes[args.in_dataset], root, preprocess, args.batch_size, False, **kwargs)
    else:
        raise NotImplementedError(f"Dataset {args.in_dataset} is not supported.")
    
    return val_loader



def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    Set OOD loader for ImageNet scale datasets
    '''    
    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    dataset_paths = {
        'iNaturalist': os.path.join(root, 'ImageNet_OOD_dataset', 'iNaturalist'),
        'SUN': os.path.join(root, 'ImageNet_OOD_dataset', 'SUN'),
        'places365': os.path.join(root, 'ImageNet_OOD_dataset', 'Places'),
        'dtd': os.path.join(root, 'ImageNet_OOD_dataset', 'dtd', 'images'),
        'ssb_hard': os.path.join(root, 'ImageNet_OOD_dataset', 'ssb_hard'),
        'ninco': os.path.join(root, 'ImageNet_OOD_dataset', 'ninco'),
        'ImageNet10': os.path.join(root, 'ImageNet10', 'val'),
        'ImageNet20': os.path.join(root, 'ImageNet20', 'val'),
        'svhn': os.path.join(root, 'svhn'),
        'lsun': os.path.join(root, 'LSUN'),
        'cifar10': os.path.join(root, 'cifar10'),
        'cifar100': os.path.join(root, 'cifar100')
    }

    custom_datasets = {
        'cub100_OOD': lambda root, transform: Cub100(root, train=False, id=False, transform=transform),
        'car98_OOD': lambda root, transform: StanfordCars98(root, split="test", id=False, transform=transform),
        'pet18_OOD': lambda root, transform: OxfordIIITPet_18(root, split="test", id=False, download=True, transform=transform),
        'food50_OOD': lambda root, transform: Food101_50(root, split="test", id=False, download=True, transform=transform)
    }

    if out_dataset in dataset_paths:
        path = dataset_paths[out_dataset]
        ood_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    elif out_dataset in custom_datasets:
        ood_loader = torch.utils.data.DataLoader(
            custom_datasets[out_dataset](root, preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    else:
        raise ValueError(f"Unknown out_dataset: {out_dataset}")

    return ood_loader