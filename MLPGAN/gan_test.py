import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# "鉴定师" - 判别器 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self,image_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 输出一个分数(logit)，而不是概率
            # 注意：这里最后没有Sigmoid激活函数
        )

    def forward(self, img):
        # 将图像扁平化成一维向量
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class Generator(nn.Module):
    def __init__(self,latent_dim,image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),  # LeakyReLU是一个常用的激活函数，防止梯度消失
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size),
            nn.Tanh()  # Tanh激活函数将输出压缩到[-1, 1], 匹配我们归一化的数据
        )

    def forward(self, z):
        img = self.model(z)
        # 将一维向量变形成图像的形状
        img = img.view(img.size(0), 1, 28, 28)
        return img

def main():
    # 1. 设置超参数和设备
    # ==================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    # 创建一个目录来保存生成的图像
    if not os.path.exists("MLPGAN/gan_images"):
        os.makedirs("MLPGAN/gan_images")

    latent_dim = 100  # 随机噪声向量的维度 (z)
    image_size = 28 * 28  # 图像大小 (784)
    batch_size = 128
    num_epochs = 50
    lr = 0.0002

    # 2. 准备数据 (MNIST数据集)
    # ==================================
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor, 并归一化到[0, 1]
        transforms.Normalize((0.5,), (0.5,))  # 将数据从[0, 1]归一化到[-1, 1]
    ])

    # 下载并加载训练数据
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    # ==================================
    generator = Generator(latent_dim,image_size).to(device)
    discriminator = Discriminator(image_size).to(device)

    # 损失函数: BCEWithLogitsLoss 包含了Sigmoid和BCELoss, 数值上更稳定
    criterion = nn.BCEWithLogitsLoss()

    # 优化器: Adam是一种高效的优化算法
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # 创建一个固定的噪声向量，用于在训练过程中可视化生成器的进步
    fixed_noise = torch.randn(64, latent_dim).to(device)

    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)

            # 创建真实标签和虚假标签
            real_labels = torch.full((real_images.size(0), 1), 0.9).to(device)  # 全部填充为0.9
            fake_labels = torch.full((real_images.size(0), 1), 0.1).to(device)  # 全部填充为0.1

            # 1. 计算真实图像的损失
            d_outputs_real = discriminator(real_images)
            d_loss_real = criterion(d_outputs_real, real_labels)

            # 2. 计算虚假图像的损失
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(z)
            # 使用.detach()来切断梯度流，因为我们只想更新判别器
            d_outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(d_outputs_fake, fake_labels)

            # 3. 判别器的总损失并更新
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()  # 梯度清零
            d_loss.backward()  # 反向传播
            d_optimizer.step()  # 更新权重

            # 生成一批新的假图像
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(z)

            # 计算生成器的损失
            # 我们希望生成器生成的图像能够被判别器认为是"真实"的(标签为1)
            g_outputs = discriminator(fake_images)
            g_loss = criterion(g_outputs, real_labels)  # 注意这里用的是real_labels

            # 更新生成器
            g_optimizer.zero_grad()  # 梯度清零
            g_loss.backward()  # 反向传播
            g_optimizer.step()  # 更新权重

            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

            # 每个epoch结束后，保存一张由固定噪声生成的图像，以观察效果
        with torch.no_grad():
            # 生成图像并取消归一化（从[-1, 1] -> [0, 1]）
            sample_images = generator(fixed_noise).cpu() * 0.5 + 0.5
            torchvision.utils.save_image(sample_images, f'MLPGAN/gan_images/epoch_{epoch + 1}.png')

    print("训练完成!")

if __name__ == '__main__':
    main()