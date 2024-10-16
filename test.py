from model import Generator, Discriminator
from dataloader import data_loader
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Trainer for GAN')
parser.add_argument('--train_list', default='testlist.txt', type=str,
                    help='训练数据列表')
parser.add_argument('--weights', default='weights/model.pth', type=str,
                    help='保存路径')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='训练轮次')
parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--device', default=0, type=int, metavar='N',
                    help='GPU')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
args = parser.parse_args()

print('构建模型......')
generator = Generator()
generator = generator.cuda(args.device)
generator.load_state_dict(torch.load(args.weights))
print('构建模型：√')

print('构建数据载入器......')
train_loader = data_loader(args)
print('构建数据载入器√')

print('构建损失函数......')
criterion = nn.BCELoss()
print('构建数损失函数√')


# 定义训练函数
def test(args):
    generator.eval()

    train_bar = tqdm(train_loader, file=sys.stdout)
    for tensor_xy, p in train_bar:
        tensor_xy = tensor_xy.cuda(args.device)

        fake_data = generator(tensor_xy)
        print("坐标:{},信号强度{}".format(tensor_xy,fake_data))

        train_bar.desc = "test  ".format()



# 训练模型
test(args)
