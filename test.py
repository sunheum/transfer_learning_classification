import argparse
import matplotlib.pyplot as plt

import torch

from classification.data_loader import get_loaders
from classification.model_loader import get_model
from classification.utils import visualize_model

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_path', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--valid_ratio', type=float, default=.2)

    p.add_argument('--model_name', type=str, default='resnet')
    p.add_argument('--dataset_name', type=str, default='catdog')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_classes', type=int, default=2)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--use_pretrained', action='store_true')

    config = p.parse_args()

    return config

def main(config):

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # 모델 Load
    model, input_size = get_model(config)
    model.load_state_dict(torch.load(config.model_path))
    model = model.to(device)
    model.eval()

    # 테스트 데이터셋 Load
    train_loader, valid_loader, test_loader = get_loaders(config, input_size)
    print("Test:", len(test_loader.dataset))

    # 시각화
    visualize_model(model, device, test_loader, num_images=6)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    config = define_argparser()
    main(config)