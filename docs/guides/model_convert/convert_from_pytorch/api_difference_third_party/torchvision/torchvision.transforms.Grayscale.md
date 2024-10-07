## [paddle 参数更多]torchvision.transforms.Grayscale

### [torchvision.transforms.Grayscale](https://pytorch.org/vision/main/generated/torchvision.transforms.Grayscale.html)

```python
torchvision.transforms.Grayscale(num_output_channels: int = 1)
```

### [paddle.vision.transforms.Grayscale](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Grayscale_cn.html)

```python
paddle.vision.transforms.Grayscale(
    num_output_channels: int = 1, 
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                     |
| -------------------------------- | ----------------------------------- | ---------------------------------------- |
| num_output_channels         | num_output_channels            | 输出图像的通道数，参数值为 1 或 3。       |
| -                                | keys | Paddle 支持 `keys` 参数。 |
