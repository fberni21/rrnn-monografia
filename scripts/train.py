import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

import numpy as np
import matplotlib.pyplot as plt


def train(model, optimizer, criterion, train_loader,
          epochs=10, test_loader=None, normalize_filters=False,
          verbose=True):
    losses = []
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

        if verbose:
            print(f'[Epoch {epoch + 1:2d}/{epochs}]',
                  f'Training loss: {running_loss:.6f}', end='')

            if test_loader is not None:
                accuracy, loss = compute_accuracy(model, test_loader,
                                                  loss_criterion=criterion)
                losses.append([running_loss, loss])
                print(f' - Test accuracy: {accuracy:.2f} %', end='')
            else:
                losses.append(running_loss)

            print()
        else:
            losses.append(running_loss)

    return np.array(losses)


def compute_accuracy(model, loader, loss_criterion=None):
    hits = 0
    model.eval()
    dataset_len = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            if loss_criterion is not None:
                running_loss += loss_criterion(outputs, labels).item()
            _, predictions = torch.max(outputs.data, 1)
            hits += (predictions == labels).sum().item()
            dataset_len += labels.shape[0]

    accuracy = 100 * hits / dataset_len
    if loss_criterion is not None:
        return accuracy, (running_loss / dataset_len)
    else:
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
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight
            rms = torch.sqrt(torch.mean(w**2, dim=(1, 2, 3), keepdim=True))
            scale = max_rms / torch.clamp(rms, min=max_rms)
            module.weight.mul_(scale)


def main():
    from torch import nn, optim
    import models

    torch.manual_seed(1131615)

    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    MODEL = 'large'

    if MODEL == 'first':
        model = models.FirstNet()
        path = '../weights/firstnet_fashion.pth'
        normalize_filters = False
        lr = 0.001
    elif MODEL == 'second':
        model = models.SecondNet()
        path = '../weights/secondnet_fashion.pth'
        normalize_filters = True
        lr = 0.001
    elif MODEL == 'third':
        model = models.ThirdNet()
        path = '../weights/thirdnet_fashion.pth'
        normalize_filters = True
        lr = 0.002
    elif MODEL == 'large':
        model = models.LargeNet()
        path = '../weights/largenet_fashion.pth'
        normalize_filters = True
        lr = 0.001
    else:
        raise ValueError(f"Model '{MODEL}' is not valid")

    num_parameters = sum([p.numel()
                          for p in model.parameters() if p.requires_grad])
    print(f'Model `{MODEL}`  has {num_parameters} parameters.')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    losses = train(model, optimizer, criterion, train_loader,
                   epochs=epochs, test_loader=test_loader,
                   normalize_filters=normalize_filters, verbose=True)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, epochs+1), losses[:, 0], label='Train')
    plt.plot(np.arange(1, epochs+1), losses[:, 1], label='Test')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../img/{MODEL}_loss.svg')
    plt.show()

    torch.save(model.state_dict(), path)
    print(f"Test accuracy: {compute_accuracy(model, test_loader)} %")


if __name__ == '__main__':
    main()
