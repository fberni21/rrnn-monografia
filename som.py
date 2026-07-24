import matplotlib.pyplot as plt
import minisom
import numpy as np
import torch

import models
from train import load_data


def visualize_som_examples(x, wmap, grid_size, save_path=None):
    image = np.zeros((grid_size * 28, grid_size * 28))

    for j in range(grid_size):
        for i in range(grid_size):
            if (i, j) in wmap:
                image[28 * j:28 * (j + 1), 28 * i:28 * (i + 1)] = x[wmap[(i, j)], :, ::-1]

    plt.figure(figsize=(30, 30))
    plt.imshow(image, cmap='Greys', interpolation='nearest', origin='lower')
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def train_som(data, grid_size, sigma=1.0, learning_rate=1.0,
              num_iteration=1000):
    som = minisom.MiniSom(grid_size, grid_size, data.shape[1], sigma=sigma,
                          learning_rate=learning_rate, random_seed=1131615)
    som.train(data, num_iteration=num_iteration, verbose=True)

    wmap = {}
    im = 0
    for x in data:
        w = som.winner(x)
        wmap[w] = im
        im += 1

    return som, wmap


def visualize_som_winners(som, data, targets, save_path=None):
    plt.figure(figsize=(12, 12))

    for x, t in zip(data, targets):
        w = som.winner(x)
        plt.text(w[0]+.5,  w[1]+.5,  str(int(t)),
                 color=plt.cm.rainbow(t / 10.), fontdict={'size': 9})

    plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def main():
    torch.manual_seed(1131615)
    np.random.seed(1131615)

    MODEL_PARAMS = {
            'first': {
                'layer': 4,
                'type': models.FirstNet,
                'pth_file': 'firstnet_fashion.pth',
                },
            'second': {
                'layer': 9,
                'type': models.SecondNet,
                'pth_file': 'secondnet_fashion.pth',
                },
            }

    with torch.no_grad():
        loader = load_data(batch_size=10000, train=False)

        x, y = next(iter(loader))

        grid_size = 80

        for model_name, params in MODEL_PARAMS.items():
            print(model_name)

            model = params['type'](pth_file=params['pth_file'])
            model.eval()
            output = model(x)
            _, pred = torch.max(output.data, 1)

            data = model.activations[params['layer']]
            data = data.view(data.shape[0], -1).numpy()

            som, wmap = train_som(data, grid_size, sigma=5.0,
                                  learning_rate=1.0, num_iteration=10000)

            visualize_som_winners(som, data, y,
                                  save_path=f'{model_name}_som.svg')

            visualize_som_examples(x.numpy(), wmap, grid_size,
                                   save_path=f'{model_name}_som_examples.svg')

    input()


if __name__ == '__main__':
    main()

# vim: ts=4 sts=4 sw=4
