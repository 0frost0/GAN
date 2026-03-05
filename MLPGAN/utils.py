import numpy as np
from PIL import Image
import torch


def generate_morph_gif(generator, latent_dim, device):
    print("正在生成数字变形 GIF...")

    # 1. 随机生成两个“种子” (起点和终点)
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)

    # 2. 生成 60 帧的过渡 (插值)
    # 我们在 z1 和 z2 之间根据比例 alpha 慢慢移动
    # alpha 从 0 变到 1
    frames = []
    steps = 60

    generator.eval()  # 切换到评估模式

    with torch.no_grad():
        # 首先生成起点和终点的数字看看（可选）
        img1 = generator(z1).cpu().squeeze().numpy()
        img2 = generator(z2).cpu().squeeze().numpy()

        for i in range(steps):
            alpha = i / (steps - 1)
            # 核心公式：线性插值 (Linear Interpolation)
            # z_new = (1 - alpha) * z1 + alpha * z2
            z_interp = (1 - alpha) * z1 + alpha * z2

            # 让生成器画图
            fake_img = generator(z_interp)

            # 处理图片格式以保存为 GIF
            # 1. 去掉 batch 维度 -> (1, 28, 28)
            # 2. 反归一化 (-1~1 -> 0~1) -> * 0.5 + 0.5
            # 3. 转成 0-255 的整数
            img_tensor = fake_img.squeeze() * 0.5 + 0.5
            img_numpy = (img_tensor.cpu().numpy() * 255).astype(np.uint8)

            # 变成 PIL Image 对象
            im = Image.fromarray(img_numpy)
            # 放大一点，不然 28x28 太小了看不清
            im = im.resize((280, 280), resample=Image.NEAREST)
            frames.append(im)

    # 3. 保存 GIF
    save_path = "digit_morph.gif"
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"GIF 已保存为: {save_path}，快去打开看看！")

# ==========================================
# 如何使用：
# 在你的 main() 函数最后，训练完成后调用它：
# generate_morph_gif(generator, latent_dim, device)
# ==========================================