import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from model import Generator, Discriminator
from dataloader import data_loader

parser = argparse.ArgumentParser(description='Trainer for GAN')
parser.add_argument('--train_list', default='trainlist.txt', type=str,
                    help='训练数据列表')
parser.add_argument('--save_path', default='weights', type=str,
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
discriminator = Discriminator()
generator = generator.cuda(args.device)
discriminator = discriminator.cuda(args.device)
print('构建模型：√')

print('构建优化器......')
g_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
print('构建优化器√')

print('构建数据载入器......')
train_loader = data_loader(args)
print('构建数据载入器√')

print('构建损失函数......')
criterion = nn.BCELoss()
print('构建数损失函数√')


# 定义训练函数
def train(args):
    generator.train()
    discriminator.train()

    train_bar = tqdm(train_loader, file=sys.stdout)
    for epoch in range(args.epochs):
        loss_g = 0
        loss_d = 0
        trained_size = 0
        for tensor_xy, p in train_bar:
            tensor_xy = tensor_xy.cuda(args.device)
            p = p.cuda(args.device)

            # 训练判别器
            d_optimizer.zero_grad()
            real_data = p
            fake_data = generator(tensor_xy).detach()
            real_predict = discriminator(real_data)
            fake_predict = discriminator(fake_data)
            d_loss = criterion(real_predict, torch.ones_like(real_predict)) + criterion(fake_predict,
                                                                                        torch.zeros_like(fake_predict))

            loss_d += d_loss.item() * p.size(0)
            trained_size += p.size(0)

            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            fake_data = generator(tensor_xy)
            fake_predict = discriminator(fake_data)
            g_loss = criterion(fake_predict, torch.ones_like(fake_predict))

            loss_g += g_loss.item() * p.size(0)

            g_loss.backward()
            g_optimizer.step()

            avg_loss_d = loss_d / trained_size
            avg_loss_g = loss_g / trained_size

            train_bar.desc = "train epoch[{}/{}] loss_g:{:.3f} loss_d:{:.3f}".format(epoch + 1, args.epochs,
                                                                                     avg_loss_g, avg_loss_d)

        model_save_path = "model" + ".pth"
        model_save_path = os.path.join(args.save_path, model_save_path)
        print(("saving..."))
        torch.save(generator.state_dict(), model_save_path)





# 训练模型
train(args)
