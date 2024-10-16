## [输入参数用法不一致]torchvision.datasets.CIFAR10

### [torchvision.datasets.CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

```python
torchvision.datasets.CIFAR10(root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

### [paddle.vision.datasets.Cifar10](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/Cifar10_cn.html)

```python
paddle.vision.datasets.Cifar10(data_file: Optional[str] = None, mode: str = 'train', transform: Optional[Callable] = None, download: bool = True, backend: Optional[str] = None)
```

两者功能一致，指定数据集文件路径的参数 `root` 与指定训练集的参数 `train` 的用法不一致，具体如下：

### 参数映射

| torchvision        | PaddlePaddle           | 备注                                                       |
| ---------------------- | --------------------- | ---------------------------------------------------------- |
| root                   | data_file             | 数据集文件路径，Paddle 参数 data_file 需含完整的文件名，如 PyTorch 参数 `/path/to/data`，对应 Paddle 参数 `/path/to/data/cifar-10-python.tar.gz`，需要转写。         |
| train                  | mode                  | 训练集或者数据集。PyTorch 参数 train=True 对应 Paddle 参数 mode='train'，PyTorch 参数 train=False 对应 Paddle 参数 mode='test'，需要转写。 |
| transform              | transform             | 图片数据的预处理。           |
| target_transform       | -                     | 接受目标数据并转换，Paddle 无此参数，暂无转写方式。    |
| download               | download              | 是否自动下载数据集文件，参数默认值不一致。PyTorch 默认为 False，Paddle 默认为 True，Paddle 需设置为与 PyTorch 一致。 |
| -                      | backend               | 指定图像类型，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例
#### root：数据集文件路径
```python
# PyTorch 写法
train_dataset = torchvision.datasets.CIFAR10(root='/path/to/data', train=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.Cifar10(data_file=str(pathlib.Path('/path/to/data') / 'cifar-10-python.tar.gz'), mode='train')
```

#### train: 训练集或数据集
训练集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.CIFAR10(train=True, download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.Cifar10(mode='train', download=True)
```

测试集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.CIFAR10(train=False, download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.Cifar10(mode='test', download=True)
```
