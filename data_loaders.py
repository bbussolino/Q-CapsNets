import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_mnist(args):
    """Load MNIST dataset.
    Data are split between train and test sets.
    Args: 
        args: arguments of the main 
    Returns: 
        training_data_loader: data loader of the training set 
        testing_data_loader: data loader of the testing set 
        num_channels: number of channels of the images 
        wh: height/width of the images in pixels 
        num_classes: number of classes of the dataset 
    """
    # Samples of the training set are randomly translated of two pixels
    data_transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2 / 28, 2 / 28), scale=None, shear=None, resample=False,
                                fillcolor=0),
        transforms.ToTensor()
    ])

    data_transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': False} if torch.cuda.device_count() > 0 else {}

    print('===> Loading MNIST training datasets')
    # MNIST training dataset
    training_set = datasets.MNIST(
        '../data', train=True, download=True, transform=data_transform_train)
    # Training data loader
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading MNIST testing datasets')
    # MNIST testing dataset
    testing_set = datasets.MNIST(
        '../data', train=False, download=True, transform=data_transform_test)
    # Testing data loader
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    num_channels = 1
    num_classes = 10
    wh = 28  # Horizontal/vertical dimension of the images

    return training_data_loader, testing_data_loader, num_channels, wh, num_classes


def load_fmnist(args):
    """Load FashionMNIST dataset.
    Data are split between train and test sets.
    Args: 
        args: arguments of the main 
    Returns: 
        training_data_loader: data loader of the training set 
        testing_data_loader: data loader of the testing set 
        num_channels: number of channels of the images 
        wh: height/width of the images in pixels 
        num_classes: number of classes of the dataset 
    """
    # Samples of the training set are randomly translated of two pixels and flipped horizontally 
    # with 0.2 probability
    data_transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2 / 28, 2 / 28), scale=None, shear=None, resample=False,
                                fillcolor=0),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor()
    ])

    data_transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': False} if torch.cuda.device_count() > 0 else {}

    print('===> Loading Fashion_MNIST training datasets')
    # FashionMNIST training dataset
    training_set = datasets.FashionMNIST(
        './data/FMNIST', train=True, download=True, transform=data_transform_train)
    # Training data loader
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading Fashion_MNIST testing datasets')
    # FashionMNIST testing dataset
    testing_set = datasets.FashionMNIST(
        './data/FMNIST', train=False, download=True, transform=data_transform_test)
    # Testing data loader
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    num_channels = 1
    num_classes = 10
    wh = 28  # Horizontal/vertical dimension of the images

    return training_data_loader, testing_data_loader, num_channels, wh, num_classes


def load_cifar10(args):
    """Load CIFAR10 dataset.
    Data are split between train and test sets.
    Args: 
        args: arguments of the main 
    Returns: 
        training_data_loader: data loader of the training set 
        testing_data_loader: data loader of the testing set 
        num_channels: number of channels of the images 
        wh: height/width of the images in pixels 
        num_classes: number of classes of the dataset 
    """
    # Samples of the training set are randomly translated of five pixels and flipped horizontally
    # with 0.5 probability
    data_transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=2, translate=(5 / 64, 5 / 64), scale=None, shear=None, resample=False,
                                fillcolor=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': False} if torch.cuda.device_count() > 0 else {}

    print('===> Loading CIFAR10 training datasets')
    # CIFAR10 training dataset
    training_set = datasets.CIFAR10(
        './data/CIFAR10', train=True, download=True, transform=data_transform_train)
    # Training data loader
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading CIFAR10 testing datasets')
    # CIFAR10 testing dataset 
    testing_set = datasets.CIFAR10(
        './data/CIFAR10', train=False, download=True, transform=data_transform_test)
    # Testing data loader 
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    num_channels = 3
    num_classes = 10
    wh = 64  # Horizontal/vertical dimension of the images

    return training_data_loader, testing_data_loader, num_channels, wh, num_classes


def load_svhn(args):
    """Load SVHN dataset.
    The data is split and normalized between train and test sets.
    Args: 
        args: arguments of the main 
    Returns: 
        training_data_loader: data loader of the training set 
        testing_data_loader: data loader of the testing set 
        num_channels: number of channels of the images 
        wh: height/width of the images in pixels 
        num_classes: number of classes of the dataset 
    """
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': False} if torch.cuda.device_count() > 0 else {}

    print('===> Loading SVHN training datasets')
    # SVHN training dataset
    training_set = datasets.SVHN(
        './data/SVHN', train=True, download=True, transform=data_transform)
    # Training data loader 
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading SVHN testing datasets')
    # SVHN testing dataset 
    testing_set = datasets.SVHN(
        './data/SVHN', train=False, download=True, transform=data_transform)
    # Testing data loader 
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    num_channels = 3
    num_classes = 10
    wh = 64  # Horizontal/vertical dimension of the images

    return training_data_loader, testing_data_loader, num_channels, wh, num_classes
