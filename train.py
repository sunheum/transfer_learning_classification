import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from classification.data_loader import get_loaders
from classification.trainer import Trainer
from classification.model_loader import get_model
from classification.utils import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/transfer_learning_experiment')

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model_name', type=str, default='resnet')
    p.add_argument('--dataset_name', type=str, default='catdog')
    p.add_argument('--n_classes', type=int, default=2)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--use_pretrained', action='store_true')

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    if config.verbose >= 2:
        print(config)

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    model, input_size = get_model(config)
    model = model.to(device)

    train_loader, valid_loader, test_set = get_loaders(config, input_size)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))

    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(config.n_epochs):
        print('epoch : {} / {}'.format(epoch+1, config.n_epochs))
        # Train
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for i, mini_batch in enumerate(train_loader, 0):
            x, y = mini_batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            _, preds = torch.max(y_hat, 1)
            loss = crit(y_hat, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(preds == y.data)

        loss = running_loss / len(train_loader.dataset)
        accuracy = running_corrects.double() / len(train_loader.dataset)

        writer.add_scalar('Train/Loss', loss, epoch+1)
        writer.add_scalar('Train/Accuracy', accuracy, epoch+1)
        print(f'Train Loss : {loss:.4f} / Train Accuracy : {accuracy:.4f}')
        
        # Valid
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        for i, mini_batch in enumerate(valid_loader, 0):
            model.eval()
            with torch.no_grad():
                x, y = mini_batch
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                y_hat = model(x)
                _, preds = torch.max(y_hat, 1)
                loss = crit(y_hat, y)

                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data)

        loss = running_loss / len(valid_loader.dataset)
        accuracy = running_corrects.double() / len(valid_loader.dataset)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        writer.add_scalar('Validation/Loss', loss, epoch+1)
        writer.add_scalar('Validation/Accuracy', accuracy, epoch+1)
        print(f'Validation Loss : {loss:.4f} / Validation Accuracy : {accuracy:.4f}')

    writer.close()
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
