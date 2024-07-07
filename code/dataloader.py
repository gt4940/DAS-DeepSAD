from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def transform_label(label):
    if label == 1:
        return -1
    if label == 0:
        return 1
    if label == 2:
        return 0


def make_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, data_transform, transform_label)
    # print(dataset.find_classes(data_dir))
    dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.num_workers_dataloader)
    return dataloader


def make_augmented_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(),
                                         transforms.RandomApply([transforms.RandomVerticalFlip(0.2),
                                                                 transforms.RandomHorizontalFlip(0.2),
                                                                 transforms.RandomInvert(0.5),
                                                                 transforms.RandomAutocontrast()]),
                                         transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, data_transform, transform_label)
    # print(dataset.find_classes(data_dir))
    dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.num_workers_dataloader, drop_last=True)
    return dataloader
