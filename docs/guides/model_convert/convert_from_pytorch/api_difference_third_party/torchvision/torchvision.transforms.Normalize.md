## [torch 参数更多]torchvision.transforms.Normalize

### [torchvision.transforms.Normalize](https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

```python
torchvision.transforms.Normalize(
    mean: Union[List[float], Tuple[float, ...]],
    std: Union[List[float], Tuple[float, ...]],
    inplace: bool = False
)
```

### [paddle.vision.transforms.Normalize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Normalize__upper_cn.html#normalize)

```python
paddle.vision.transforms.Normalize(
    mean: Union[int, float, List[float], Tuple[float, ...]] = 0.0,
    std: Union[int, float, List[float], Tuple[float, ...]] = 1.0,
    data_format: str = 'CHW',
    to_rgb: bool = False,
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| mean                   | mean  | 用于每个通道归一化的均值。                                   |
| std                    | std   | 用于每个通道归一化的标准差值。                               |
| inplace          | -                                     | Paddle 不支持 `inplace` 参数。                               |
| -                                | data_format                      | Paddle 支持 `data_format` 参数，用于指定数据格式。默认为 'CHW'。 |
| -                                | to_rgb                          | Paddle 支持 `to_rgb` 参数，是否将图像转换为 RGB 格式。默认为 False。 |
| -                                | keys         | Paddle 支持 `keys` 参数，默认为 None。 |

### 转写示例

#### 基本归一化

```python
# PyTorch 写法
import torch
import torchvision.transforms as transforms
img = torch.tensor([
    [[0.5, 0.5], [0.5, 0.5]],
    [[0.5, 0.5], [0.5, 0.5]],
    [[0.5, 0.5], [0.5, 0.5]]
])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_img = normalize(img)

# Paddle 写法
import paddle
img = paddle.to_tensor(data=[[[0.5, 0.5], [0.5, 0.5]],
                             [[0.5, 0.5], [0.5, 0.5]],
                             [[0.5, 0.5], [0.5, 0.5]]])
normalize = paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_img = normalize(img)

```
