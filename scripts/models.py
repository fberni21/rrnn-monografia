import torch
from torch import nn


class FirstNet(nn.Module):
    def __init__(self, pth_file=None):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 5, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, return_indices=True),
                nn.Conv2d(5, 15, kernel_size=7, stride=2, padding=3),
                nn.ReLU())

        self.full = nn.Sequential(
                nn.Flatten(),
                nn.Linear(15 * 4 * 4, 84),
                nn.ReLU(),
                nn.Linear(84, 10))

        self.pool_indices = dict()
        self.activations = dict()

        if pth_file is not None:
            self._initialize_weights(pth_file)

    def forward(self, x):
        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.MaxPool2d):
                x, m = layer(x)
                self.pool_indices[i] = m
            else:
                x = layer(x)

            self.activations[i] = x

        x = self.full(x)
        return x

    def _initialize_weights(self, pth_file):
        state_dict = torch.load(pth_file, weights_only=True)

        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = state_dict[f'conv.{i}.weight'].data
                layer.bias.data = state_dict[f'conv.{i}.bias'].data

        for i, layer in enumerate(self.full):
            if isinstance(layer, nn.Linear):
                layer.weight.data = state_dict[f'full.{i}.weight'].data
                layer.bias.data = state_dict[f'full.{i}.bias'].data


class DeconvFirstNet(nn.Module):
    def __init__(self, pth_file=None):
        super().__init__()

        self._deconv2conv = {0: 3, 3: 0}
        self._conv2deconv = {0: 3, 3: 0}
        self._unpool2pool = {1: 2}

        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(15, 5, kernel_size=7, stride=2, padding=3, output_padding=0, bias=False),
                nn.MaxUnpool2d(kernel_size=2),
                nn.ReLU(),
                nn.ConvTranspose2d(5, 1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False))

        if pth_file is not None:
            self._initialize_weights(pth_file)

    def forward(self, x, layer_idx, map_idx, pool_indices):
        if layer_idx not in self._conv2deconv.keys():
            raise ValueError(f'Layer {layer_idx} is not a convolutional layer')

        start_idx = self._conv2deconv[layer_idx]

        layer = self.deconv[start_idx]
        start_layer = nn.ConvTranspose2d(1, layer.out_channels,
                                         kernel_size=layer.kernel_size,
                                         bias=False,
                                         padding=layer.padding,
                                         stride=layer.stride,
                                         output_padding=layer.output_padding)
        start_layer.weight = nn.Parameter(self.deconv[start_idx]
                                          .weight[map_idx])
        x = start_layer(x)

        for i, layer in enumerate(self.deconv[start_idx + 1:],
                                  start=start_idx + 1):
            if isinstance(layer, torch.nn.MaxUnpool2d):
                x = layer(x, pool_indices[self._unpool2pool[i]])
                x = nn.functional.relu(x)
            else:
                x = layer(x)
        return x

    def _initialize_weights(self, pth_file):
        state_dict = torch.load(pth_file, weights_only=True)

        for i, layer in enumerate(self.deconv):
            if isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data = state_dict[
                        f'conv.{self._deconv2conv[i]}.weight'].data


class SecondNet(nn.Module):
    def __init__(self, pth_file=None):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, return_indices=True),
                nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, return_indices=True),
                nn.Conv2d(16, 36, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(36, 36, kernel_size=3, stride=1),
                nn.ReLU())

        self.full = nn.Sequential(
                nn.Flatten(),
                nn.Linear(36 * 3 * 3, 10))

        self.pool_indices = dict()
        self.activations = dict()

        if pth_file is not None:
            self._initialize_weights(pth_file)

    def forward(self, x):
        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.MaxPool2d):
                x, m = layer(x)
                self.pool_indices[i] = m
            else:
                x = layer(x)

            self.activations[i] = x

        x = self.full(x)
        return x

    def _initialize_weights(self, pth_file):
        state_dict = torch.load(pth_file, weights_only=True)

        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = state_dict[f'conv.{i}.weight'].data
                layer.bias.data = state_dict[f'conv.{i}.bias'].data

        for i, layer in enumerate(self.full):
            if isinstance(layer, nn.Linear):
                layer.weight.data = state_dict[f'full.{i}.weight'].data
                layer.bias.data = state_dict[f'full.{i}.bias'].data


class DeconvSecondNet(nn.Module):
    def __init__(self, pth_file=None):
        super().__init__()

        self._deconv2conv = {0: 8, 2: 6, 5: 3, 8: 0}
        self._conv2deconv = {0: 8, 3: 5, 6: 2, 8: 0}
        self._unpool2pool = {3: 5, 6: 2}

        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(36, 36, kernel_size=3, stride=1, padding=0, output_padding=0, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(36, 16, kernel_size=3, stride=1, padding=0, output_padding=0, bias=False),
                nn.MaxUnpool2d(kernel_size=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 9, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
                nn.MaxUnpool2d(kernel_size=2),
                nn.ReLU(),
                nn.ConvTranspose2d(9, 1, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False))

        if pth_file is not None:
            self._initialize_weights(pth_file)

    def forward(self, x, layer_idx, map_idx, pool_indices):
        if layer_idx not in self._conv2deconv.keys():
            raise ValueError(f'Layer {layer_idx} is not a convolutional layer')

        start_idx = self._conv2deconv[layer_idx]

        layer = self.deconv[start_idx]
        start_layer = nn.ConvTranspose2d(1, layer.out_channels,
                                         kernel_size=layer.kernel_size,
                                         bias=False,
                                         padding=layer.padding,
                                         stride=layer.stride,
                                         output_padding=layer.output_padding)
        start_layer.weight = nn.Parameter(self.deconv[start_idx]
                                          .weight[map_idx])
        x = start_layer(x)

        for i, layer in enumerate(self.deconv[start_idx + 1:],
                                  start=start_idx + 1):
            if isinstance(layer, torch.nn.MaxUnpool2d):
                x = layer(x, pool_indices[self._unpool2pool[i]])
                x = nn.functional.relu(x)
            else:
                x = layer(x)
        return x

    def _initialize_weights(self, pth_file):
        state_dict = torch.load(pth_file, weights_only=True)

        for i, layer in enumerate(self.deconv):
            if isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data = state_dict[
                        f'conv.{self._deconv2conv[i]}.weight'].data


class ThirdNet(nn.Module):
    def __init__(self, pth_file=None):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 9, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, return_indices=True),
                nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 25, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, return_indices=True),
                nn.Conv2d(25, 25, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(25, 36, kernel_size=3, stride=1),
                nn.ReLU())

        self.full = nn.Sequential(
                nn.Flatten(),
                nn.Linear(36 * 3 * 3, 10))

        self.pool_indices = dict()
        self.activations = dict()

        if pth_file is not None:
            self._initialize_weights(pth_file)

    def forward(self, x):
        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.MaxPool2d):
                x, m = layer(x)
                self.pool_indices[i] = m
            else:
                x = layer(x)

            self.activations[i] = x

        x = self.full(x)
        return x

    def _initialize_weights(self, pth_file):
        state_dict = torch.load(pth_file, weights_only=True)

        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = state_dict[f'conv.{i}.weight'].data
                layer.bias.data = state_dict[f'conv.{i}.bias'].data

        for i, layer in enumerate(self.full):
            if isinstance(layer, nn.Linear):
                layer.weight.data = state_dict[f'full.{i}.weight'].data
                layer.bias.data = state_dict[f'full.{i}.bias'].data


class DeconvThirdNet(nn.Module):
    def __init__(self, pth_file=None):
        super().__init__()

        self._deconv2conv = {0: 12, 2: 10, 5: 7, 7: 5, 10: 2, 12: 0}
        self._conv2deconv = {0: 12, 2: 10, 5: 7, 7: 5, 10: 2, 12: 0}
        self._unpool2pool = {3: 9, 8: 4}

        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(36, 25, kernel_size=3, stride=1, padding=0, output_padding=0, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(25, 25, kernel_size=3, stride=1, padding=0, output_padding=0, bias=False),
                nn.MaxUnpool2d(kernel_size=2),
                nn.ReLU(),
                nn.ConvTranspose2d(25, 16, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 9, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
                nn.MaxUnpool2d(kernel_size=2),
                nn.ReLU(),
                nn.ConvTranspose2d(9, 4, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False))

        if pth_file is not None:
            self._initialize_weights(pth_file)

    def forward(self, x, layer_idx, map_idx, pool_indices):
        if layer_idx not in self._conv2deconv.keys():
            raise ValueError(f'Layer {layer_idx} is not a convolutional layer')

        start_idx = self._conv2deconv[layer_idx]

        layer = self.deconv[start_idx]
        start_layer = nn.ConvTranspose2d(1, layer.out_channels,
                                         kernel_size=layer.kernel_size,
                                         bias=False,
                                         padding=layer.padding,
                                         stride=layer.stride,
                                         output_padding=layer.output_padding)
        start_layer.weight = nn.Parameter(self.deconv[start_idx]
                                          .weight[map_idx])
        x = start_layer(x)

        for i, layer in enumerate(self.deconv[start_idx + 1:],
                                  start=start_idx + 1):
            if isinstance(layer, torch.nn.MaxUnpool2d):
                x = layer(x, pool_indices[self._unpool2pool[i]])
                x = nn.functional.relu(x)
            else:
                x = layer(x)
        return x

    def _initialize_weights(self, pth_file):
        state_dict = torch.load(pth_file, weights_only=True)

        for i, layer in enumerate(self.deconv):
            if isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data = state_dict[
                        f'conv.{self._deconv2conv[i]}.weight'].data
