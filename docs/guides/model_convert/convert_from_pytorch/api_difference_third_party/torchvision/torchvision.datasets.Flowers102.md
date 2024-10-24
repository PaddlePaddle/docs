## [输入参数用法不一致]torchvision.datasets.Flowers102

### [torchvision.datasets.Flowers102](https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html)

```python
torchvision.datasets.Flowers102(root: Union[str, Path], split: str = 'train', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

### [paddle.vision.datasets.Flowers](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/Flowers_cn.html)

```python
paddle.vision.datasets.Flowers(data_file=None, label_file=None, setid_file=None, mode='train', transform=None, download=True, backend=None)
```

两者功能一致，指定数据集文件路径的参数 `root` 与指定训练集的参数 `split` 的用法不一致，具体如下：

### 参数映射

| torchvision        | PaddlePaddle           | 备注                                                       |
| ---------------------- | --------------------- | ---------------------------------------------------------- |
| root                   | data_file             | 数据集文件路径，Paddle 参数 data_file 需含完整的文件名，如 PyTorch 参数 `./data`，对应 Paddle 参数 `./data/flowers-102/102flowers.tgz`，需要转写。         |
| -                      | label_file            | 标签文件路径，Paddle 参数 label_file 需含完整的文件名，如 PyTorch 参数 `./data`，对应 Paddle 参数 `./data/flowers-102/imagelabels.mat`，需要转写。         |
| -                      | setid_file            | 子数据集下标划分文件路径，Paddle 参数 setid_file 需含完整的文件名，如 PyTorch 参数 `./data`，对应 Paddle 参数 `./data/flowers-102/setid.mat`，需要转写。         |
| split                  | mode                  | 训练集、数据集或验证集。对于训练集和数据集，PyTorch 参数与 Paddle 参数相同，为 'train' 或 'test'，对于验证集，PyTorch 参数 split='val' 对应 Paddle 参数 mode='valid'，需要转写。 |
| transform              | transform             | 图片数据的预处理。           |
| target_transform       | -                     | 接受目标数据并转换，Paddle 无此参数，暂无转写方式。    |
| download               | download              | 是否自动下载数据集文件。 |
| -                      | backend               | 指定图像类型，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例
#### root：数据集文件路径
```python
# PyTorch 写法
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train')

# Paddle 写法
train_dataset = paddle.vision.datasets.Flowers102(data_file='./data/flowers-102/102flowers.tgz', label_file='./data/flowers-102/imagelabels.mat', setid_file='./data/flowers-102/setid.mat', mode='train')
```

#### split: 训练集或数据集
训练集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.Flowers102(data_file='./data/flowers-102/102flowers.tgz', label_file='./data/flowers-102/imagelabels.mat', setid_file='./data/flowers-102/setid.mat', mode='train', download=True)
```

测试集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.Flowers102(data_file='./data/flowers-102/102flowers.tgz', label_file='./data/flowers-102/imagelabels.mat', setid_file='./data/flowers-102/setid.mat', mode='test', download=True)
```

验证集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.Flowers102(root='./data', split='val', download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.Flowers102(data_file='./data/flowers-102/102flowers.tgz', label_file='./data/flowers-102/imagelabels.mat', setid_file='./data/flowers-102/setid.mat', mode='valid', download=True)
```
