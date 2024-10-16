import torch.nn as nn

#生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G = nn.Sequential(
            nn.Linear(2, 32),  # 输入为(x, y)坐标
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出为信号强度P
        )

    def forward(self, z):
        return self.G(z)

#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.D(x)

