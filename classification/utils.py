import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

def visualize_model(model, device, data_loader, num_images=6):
    class_names = ['cat','dog']
    # was_training = model.training
    # model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for _, mini_batch in enumerate(data_loader):
            x,y = mini_batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            _, preds = torch.max(y_hat, 1)

            for j in range(x.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(x.cpu().data[j])

                if images_so_far == num_images:
                    # model.train(mode=was_training)
                    return
        # model.train(mode=was_training)

def save_model(best_model, config, **kwargs):
    torch.save(
        {
            'model': best_model,
            'config': config,
            **kwargs
        }, config.model_fn
    )