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

| torchvision | PaddlePaddle | 备注                      |
| ------------ | -------------- | ---------------------- |
| mean          | mean          | 用于每个通道归一化的均值。  |
| std           | std           | 用于每个通道归一化的标准差值。  |
| inplace       | -             | 是否原地修改，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| -             | data_format   | 数据的格式，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | to_rgb        | 是否转换为 rgb 的格式，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | keys          | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。             |
