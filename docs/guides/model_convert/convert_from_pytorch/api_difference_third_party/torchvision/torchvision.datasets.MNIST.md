## [输入参数用法不一致]torchvision.datasets.MNIST

### [torchvision.datasets.MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)

```python
torchvision.datasets.MNIST(root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

### [paddle.vision.datasets.MNIST](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/MNIST_cn.html)

```python
paddle.vision.datasets.MNIST(image_path: str = None, label_path: str = None, mode: str = 'train', transform: Callable = None, download: bool = True, backend: str = None)
```


### 参数映射

| torchvision | PaddlePaddle | 备注 |
| -------------------------------- | ---------------------------------- | ---- |
| `root`                           | `image_path`, `label_path`         | torchvision 使用 `root` 指定数据集根目录，Paddle 分别使用 `image_path` 和 `label_path` 指定图像和标签路径。|
| `train`                          | `mode`                              | torchvision 的 `train=True` 对应 Paddle 的 `mode='train'`，`train=False` 对应 `mode='test'`。|
| `transform`                      | `transform`                         | 参数一致。|
| `target_transform`               | -                                    | Paddle 不支持 `target_transform` 参数。|
| `download`                       | `download`                           | 参数一致。|
| -                                | `backend`                            | Paddle 额外支持 `backend` 参数，用于指定返回的图像类型。|

### 转写示例

```python
# PyTorch 写法
import torchvision.datasets as datasets
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# Paddle 写法
from pathlib import Path
import paddle
transform = paddle.vision.transforms.Compose(transforms=[paddle.vision.
    transforms.ToTensor(), paddle.vision.transforms.Normalize(mean=(0.1307,
    ), std=(0.3081,))])
train_dataset = paddle.vision.datasets.MNIST(transform=transform, download=True, mode='train',
    image_path=str(Path('./data') / 'MNIST/raw/train-images-idx3-ubyte.gz'), 
    label_path=str(Path('./data') / 'MNIST/raw/train-labels-idx1-ubyte.gz'))

```
