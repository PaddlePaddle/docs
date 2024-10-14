## [输入参数用法不一致]torchvision.datasets.MNIST

### [torchvision.datasets.MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)

```python
torchvision.datasets.MNIST(root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

### [paddle.vision.datasets.MNIST](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/MNIST_cn.html)

```python
paddle.vision.datasets.MNIST(image_path: str = None, label_path: str = None, mode: str = 'train', transform: Callable = None, download: bool = True, backend: str = None)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| -------------------------------- | ---------------------------------- | ---- |
| root                   | -                     | 指定数据集根目录。|
| -                      | image_path            | 图像路径，Paddle 使用 image_path 和 label_path，等价的实现 PyTorch 的 root 的功能，需要转写。|
| -                      | label_path            | 标签路径，Paddle 使用 image_path 和 label_path，等价的实现 PyTorch 的 root 的功能，需要转写。|
| train                  | mode                  | 训练集或者数据集。PyTorch 参数 train=True 对应 Paddle 参数 mode='train'，PyTorch 参数 train=False 对应 Paddle 参数 mode='test'，需要转写。 |
| transform              | transform             | 图片数据的预处理。|
| target_transform       | -                     | 接受目标数据并转换，Paddle 无此参数，暂无转写方式。    |
| download               | download              | 是否自动下载数据集文件，参数默认值不一致。PyTorch 默认为 False，Paddle 默认为 True，Paddle 需设置为与 PyTorch 一致。 |
| -                      | backend               | 指定图像类型，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例
#### root：数据集文件路径
```python
# PyTorch 写法
train_dataset = torchvision.datasets.MNIST(root='./data')

# Paddle 写法
train_dataset = paddle.vision.datasets.MNIST(
    image_path=str(pathlib.Path('./data') / 'MNIST/raw/train-images-idx3-ubyte.gz'),
    label_path=str(pathlib.Path('./data') / 'MNIST/raw/train-labels-idx1-ubyte.gz'))
```

#### train: 训练集或数据集
训练集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.MNIST(train=True, download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.MNIST(mode='train', download=True)
```

测试集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.MNIST(train=False, download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.MNIST(mode='test', download=True)
```
