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

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算梯度惩罚项 (WGAN-GP)"""
    # 随机插值系数
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)

    # 生成插值样本
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # 计算判别器对插值样本的输出
    d_interpolates = D(interpolates)

    # 创建用于计算梯度的虚拟张量
    fake = torch.ones(real_samples.size(0), 1).to(device)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 计算梯度惩罚： (||gradient||_2 - 1)^2
    gradients = gradients.view(gradients.size(0), -1)  # [batch_size, 1*28*28]
    gradient_norm = gradients.norm(2, dim=1)  # [batch_size]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def main():
    # 1. 设置超参数和设备
    # ==================================
    latent_dim = Config.latent_dim
    nfg = Config.nfg
    nfd = Config.nfd
    nc = Config.num_channels
    lr = 0.0002
    lambda_gp = 10
    n_critic = 5
    dataroot = Config.DataRoot
    batch_size = Config.batch_size
    num_epochs = Config.num_epochs
    model_save_dir = Config.model_save_dir
    output_dir = "wgan_results"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"正在使用的设备: {device}")

    logger = get_logger('./logs_wgan')
    logger.info(f'num_epochs={num_epochs},batch_size={batch_size},lr={lr},latent_dim={latent_dim},nfg={nfg},'
                f'nfd={nfd},nc={nc},lambda_gp={lambda_gp},n_critic={n_critic}')
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

    # 优化器: Adam是一种高效的优化算法
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.0,0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0,0.9))

    # 创建一个固定的噪声向量，用于在训练过程中可视化生成器的进步
    fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)

    total_step = len(data_loader)
    print("开始训练")
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            d_loss_total = 0
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
                fake_images = generator(z).detach()
                d_optimizer.zero_grad()
                d_fake = discriminator(fake_images)
                d_real = discriminator(real_images)
                d_loss = torch.mean(d_fake) - torch.mean(d_real)
                gradient_penalty = compute_gradient_penalty(discriminator,real_images.data,fake_images.data,device)
                d_loss_total = d_loss + lambda_gp * gradient_penalty
                d_loss_total.backward()
                d_optimizer.step()
            g_optimizer.zero_grad()

            # 生成一批新的假图像
            z = torch.randn(batch_size, latent_dim,1,1).to(device)
            fake_images = generator(z)

            # 计算生成器的损失

            g_outputs = discriminator(fake_images)
            g_loss = -torch.mean(g_outputs)

            # 更新生成器
            #g_optimizer.zero_grad()  # 梯度清零
            g_loss.backward()  # 反向传播
            g_optimizer.step()  # 更新权重

            if (i + 1) % 200 == 0:
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                           f'D Loss: {d_loss_total.item():.4f}, G Loss: {g_loss.item():.4f}, '
                           f'GP: {gradient_penalty.item():.4f}, D(x): {torch.mean(d_real).item():.4f}, '
                           f'D(G(z)): {torch.mean(d_fake).item():.4f}')
            # 每个epoch结束后，保存一张由固定噪声生成的图像，以观察效果
        with torch.no_grad():
            # 生成图像并取消归一化（从[-1, 1] -> [0, 1]）
            sample_images = generator(fixed_noise).cpu() * 0.5 + 0.5
            torchvision.utils.save_image(sample_images, f'wgan_results/epoch_{epoch + 1}.png')

    print("训练完成!")
    torch.save(generator.state_dict(), f'{model_save_dir}/wgenerator_final.pth')
    torch.save(discriminator.state_dict(), f'{model_save_dir}/wdiscriminator_final.pth')

if __name__ == '__main__':
    main()