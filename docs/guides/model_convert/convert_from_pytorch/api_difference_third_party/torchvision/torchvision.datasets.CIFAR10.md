## [torch 参数更多]torchvision.datasets.CIFAR10

### [torchvision.datasets.CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

```python
torchvision.datasets.CIFAR10(root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

### [paddle.vision.datasets.Cifar10](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/Cifar10_cn.html)

```python
paddle.vision.datasets.Cifar10(data_file: Optional[str] = None, mode: str = 'train', transform: Optional[Callable] = None, download: bool = True, backend: Optional[str] = None)
```


### 参数映射

| torchvision        | PaddlePaddle           | 备注                                                       |
| ---------------------- | --------------------- | ---------------------------------------------------------- |
| root                   | data_file             | 在 Paddle 中，data_file 需要包含完整的路径和文件名。         |
| train                  | mode                  | train=True 对应 mode='train'，train=False 对应 mode='test' |
| transform              | transform             | 参数相同                                                   |
| target_transform       | -                     | Paddle 不支持 target_transform 参数                        |
| download               | download              | 参数相同，但默认值不同：torchvision 默认为 False，Paddle 默认为 True |
| -                      | backend               | Paddle 支持额外的 backend 参数，用于指定图像类型（'pil' 或 'cv2'） |

### 转写示例

```python
# torchvision 写法
import torchvision.datasets as datasets
train_dataset = datasets.CIFAR10(root='/path/to/data', train=True, transform=transform, download=True)

# Paddle 写法
import paddle
from pathlib import Path
train_dataset = paddle.vision.datasets.Cifar10(transform=transform,
    download=True, data_file=str(Path('/path/to/data') / 'cifar-10-python.tar.gz'), mode='train')
```

