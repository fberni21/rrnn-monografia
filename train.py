import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


def train(model, optimizer, criterion, train_loader,
          epochs=10, test_loader=None, normalize_filters=False):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        dataset_len = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if normalize_filters:
                renormalize_conv_filters(model)

            running_loss += loss.item()
            dataset_len += labels.shape[0]

        running_loss /= dataset_len

        print(f'[Epoch {epoch + 1:2d}/{epochs}]',
              f'Training loss: {running_loss:.6f}', end='')

        if test_loader is not None:
            accuracy = compute_accuracy(model, test_loader)
            print(f' - Test accuracy: {accuracy:.2f} %', end='')

        print()


def compute_accuracy(model, loader):
    hits = 0
    model.eval()
    dataset_len = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            hits += (predictions == labels).sum().item()
            dataset_len += labels.shape[0]

    accuracy = 100 * hits / dataset_len
    return accuracy


def load_data(batch_size=128, train=True):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.2860,), (0.3529,)),
    ])
    dataset = FashionMNIST(root='data/', train=train,
                           transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


@torch.no_grad()
def renormalize_conv_filters(model: nn.Module, max_rms: float = 0.1):
    """Renormalizes convolutional filters whose RMS exceeds `max_rms` back down to `max_rms`."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight
            rms = torch.sqrt(torch.mean(w**2, dim=(1, 2, 3), keepdim=True))
            scale = max_rms / torch.clamp(rms, min=max_rms)
            module.weight.mul_(scale)


def main():
    from torch import nn, optim
    import models

    torch.manual_seed(10406308320747545401)

    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    # model = models.LeNet5()
    # model = models.FberNet()
    model = models.FberNet2()
    num_parameters = sum([p.numel()
                          for p in model.parameters() if p.requires_grad])
    print(f'Model has {num_parameters} parameters.')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, optimizer, criterion, train_loader,
          epochs=10, test_loader=test_loader, normalize_filters=True)

    # torch.save(model.state_dict(), 'lenet5_fashion.pth')
    # torch.save(model.state_dict(), 'fbernet_fashion.pth')
    torch.save(model.state_dict(), 'fbernet2_fashion.pth')


if __name__ == '__main__':
    main()
