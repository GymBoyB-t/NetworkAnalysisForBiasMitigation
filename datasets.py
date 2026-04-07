import torch
import torchvision
import torchvision.transforms as transforms




def loadMNIST():

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False
    )

    images, labels = next(iter(loader))
    images = images.numpy()
    labels = labels.numpy()

    return images, labels