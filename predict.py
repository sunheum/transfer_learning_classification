import argparse
import matplotlib.pyplot as plt
import pandas as pd

import torch

from classification.data_loader import get_loaders
from classification.model_loader import get_model
from classification.utils import imshow

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

    # 예측 데이터셋 Load
    train_loader, valid_loader, predict_loader = get_loaders(config, input_size)
    print("Predict:", len(predict_loader.dataset))

    class_names = ['cat','dog']
    dir_list = []
    predict_list = []

    with torch.no_grad():
        allFiles, _ = map(list, zip(*predict_loader.dataset.samples))
        for i, mini_batch in enumerate(predict_loader):
            x,y = mini_batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            _, preds = torch.max(y_hat, 1)

            # 예측값 저장
            for j in range(x.size()[0]):
                dir_list.append(allFiles[i*config.batch_size+j])
                predict_list.append(class_names[preds[j]])

            # 시각화
            if i==0:
                images_so_far = 0
                for j in range(x.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(3, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(x.cpu().data[j])
                    if images_so_far == 6:
                        break
                plt.ioff()
                plt.show()

    result = pd.DataFrame(zip(dir_list,predict_list), columns=['file_dir','predict'])
    result.to_csv('result.csv', encoding='utf-8-sig', index=False)
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)