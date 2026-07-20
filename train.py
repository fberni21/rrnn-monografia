import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


def train(model, optimizer, criterion, train_loader,
          epochs=10, test_loader=None):
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


def load_data(train=True):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32),
        v2.Normalize(mean=[0], std=[1])
    ])
    dataset = FashionMNIST(root='data/', train=train,
                           transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=train)
    return loader


def main():
    from torch import nn, optim
    import models

    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    torch.manual_seed(10406308320747545401)

    # model = models.LeNet5()
    model = models.FberNet()
    num_parameters = sum([p.numel()
                          for p in model.parameters() if p.requires_grad])
    print(f'Model has {num_parameters} parameters.')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, optimizer, criterion, train_loader,
          epochs=10, test_loader=test_loader)

    torch.save(model.state_dict(), 'fbernet_fashion.pth')


if __name__ == '__main__':
    main()
