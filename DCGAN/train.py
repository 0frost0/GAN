import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
from utils import get_logger

from model import Discriminator, Generator
from config import Config

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1:
        nn.init.normal_(m.weight.data,0.0,0.02)

def main():
    # 1. 设置超参数和设备
    # ==================================
    latent_dim = Config.latent_dim
    nfg = Config.nfg
    nfd = Config.nfd
    nc = Config.num_channels
    lr = Config.learning_rate
    #beta1 = Config.beta1
    dataroot = Config.DataRoot
    batch_size = Config.batch_size
    num_epochs = Config.num_epochs
    model_save_dir = Config.model_save_dir
    output_dir = Config.output_dir


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"正在使用的设备: {device}")

    logger = get_logger('./logs')
    logger.info(f'num_epochs={num_epochs},batch_size={batch_size},lr={lr},latent_dim={latent_dim},nfg={nfg},'
                f'nfd={nfd},nc={nc}')
    # 创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 准备数据 (MNIST数据集)
    # ==================================
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor, 并归一化到[0, 1]
        transforms.Normalize((0.5,), (0.5,))  # 将数据从[0, 1]归一化到[-1, 1]
    ])

    # 下载并加载训练数据
    train_dataset = torchvision.datasets.MNIST(root=dataroot, train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 初始化模型、损失函数和优化器
    # ==================================
    generator = Generator(latent_dim,nfg,nc).to(device)
    discriminator = Discriminator(nfd,nc).to(device)

    # 损失函数: BCEWithLogitsLoss 包含了Sigmoid和BCELoss, 数值上更稳定
    criterion = nn.BCEWithLogitsLoss()

    # 优化器: Adam是一种高效的优化算法
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # 创建一个固定的噪声向量，用于在训练过程中可视化生成器的进步
    fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)

    total_step = len(data_loader)
    print("开始训练")
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)

            # 创建真实标签和虚假标签
            real_labels = torch.full((batch_size, ), 0.9).to(device)  # 全部填充为0.9
            fake_labels = torch.full((batch_size, ), 0.0).to(device)  # 全部填充为0.1

            # 1. 计算真实图像的损失
            d_optimizer.zero_grad()
            d_outputs_real = discriminator(real_images).view(-1)
            d_loss_real = criterion(d_outputs_real, real_labels)
            d_loss_real.backward()

            # 2. 计算虚假图像的损失
            z = torch.randn(batch_size, latent_dim,1,1).to(device)
            fake_images = generator(z)
            # 使用.detach()来切断梯度流，因为我们只想更新判别器
            d_outputs_fake = discriminator(fake_images.detach()).view(-1)
            d_loss_fake = criterion(d_outputs_fake, fake_labels)
            d_loss_fake.backward()

            # 3. 判别器的总损失并更新

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()  # 更新权重

            g_optimizer.zero_grad()
            # 生成一批新的假图像
            z = torch.randn(batch_size, latent_dim,1,1).to(device)
            fake_images = generator(z)

            # 计算生成器的损失

            g_outputs = discriminator(fake_images).view(-1)
            g_loss = criterion(g_outputs, real_labels)

            # 更新生成器
            #g_optimizer.zero_grad()  # 梯度清零
            g_loss.backward()  # 反向传播
            g_optimizer.step()  # 更新权重

            if (i + 1) % 200 == 0:
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
            # 每个epoch结束后，保存一张由固定噪声生成的图像，以观察效果
        with torch.no_grad():
            # 生成图像并取消归一化（从[-1, 1] -> [0, 1]）
            sample_images = generator(fixed_noise).cpu() * 0.5 + 0.5
            torchvision.utils.save_image(sample_images, f'dcgan_results/epoch_{epoch + 1}.png')

    print("训练完成!")
    torch.save(generator.state_dict(), f'{model_save_dir}/generator_final.pth')
    torch.save(discriminator.state_dict(), f'{model_save_dir}/discriminator_final.pth')

if __name__ == '__main__':
    main()