# 海光 DCU 基于框架的使用指南

## 一、环境准备

### 环境说明

* 本教程介绍如何基于海光 DCU 进行 ResNet50 的训练，总共需要 1 卡进行训练

* 考虑到环境差异性，我们推荐使用教程提供的标准镜像完成环境准备：

  * 镜像链接： registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310

### 环境安装

安装 PaddlePaddle

*该命令会自动安装飞桨主框架每日自动构建的 nightly-build 版本*

*由于 dcu 代码位于飞桨主框架中，因此我们不需要安装额外的 Custom Device 包*

```shell
python -m pip install --pre paddlepaddle-dcu -i https://www.paddlepaddle.org.cn/packages/nightly/dcu/
```

## 二、运行示例

飞桨框架集成了经典的视觉模型用于帮助用户快速上手，我们将基于 ResNet50 结构，在 Cifar10 数据集上进行一次快速训练，用于帮助您了解如何基于海光 DCU 进行训练

注意：

* *本教程主要用于快速入门，并未对参数进行细致调优，训练效果未必是最好的，您可以自行调整超参数进行效果调优*

* *本教程预计使用单卡 DCU 训练 30 分钟*

1. 导入必要的包

```python
import paddle
from paddle.vision import transforms
from paddle.vision.models import resnet50
```

2. 设置运行设备

```python
# 注意：为了保持和 GPU 兼容，可以尽量少的修改模型代码，我们在命名上做了兼容处理，dcu 设备的名字同样叫做 gpu
# 1. 设定运行设备为 gpu
paddle.set_device("gpu")
```

3. 加载训练数据集

```python
# 2. 定义数据集、数据预处理方法与 DataLoader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(train_set, batch_size=128, num_workers=8)
```

4. 定义网络结构和损失函数

```python
# 3. 定义网络结构
net = resnet50(num_classes=10)
# 4. 定义损失函数
net_loss = paddle.nn.CrossEntropyLoss()
# 5. 定义优化器
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
```

5. 启动训练

训练过程中会打印 loss 的变化情况，可以观察到 loss 在初步下降，这意味着模型参数逐渐适应了该数据集。

```python
net.train()
for epoch in range(10):
    for batch_idx, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.clear_grad()
        # 6. 前向传播并计算损失
        outputs = net(inputs)
        loss = net_loss(outputs, labels)
        # 7. 反向传播
        loss.backward()
        # 8. 更新参数
        optimizer.step()
        print('Epoch %d, Iter %d, Loss: %.5f' % (epoch + 1, batch_idx + 1, loss))
print('Finished Training')
```

6. 测试模型效果

```python
test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform)

# 测试 5 张图片效果
for i in range(5):
    test_image, gt = test_dataset[0]
    # CHW -> NCHW
    test_image = test_image.unsqueeze(0)

    # 取预测分布中的最大值
    res = net(test_image).argmax().numpy()
    print(f"图像{i} 标签：{gt}")
    print(f"模型预测结果：{res}")
```
