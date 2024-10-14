## [paddle 参数更多]torchvision.transforms.RandomVerticalFlip

### [torchvision.transforms.RandomVerticalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html)

```python
torchvision.transforms.RandomVerticalFlip(p: float = 0.5)
```

### [paddle.vision.transforms.RandomVerticalFlip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomVerticalFlip_cn.html)

```python
paddle.vision.transforms.RandomVerticalFlip(
    prob: float = 0.5,
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

Paddle 比 PyTorch 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------| ------------ | ------------------------------------------------------------ |
| p           | prob         | 翻转概率，仅参数名不一致。                                       |
| -           | keys         | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。             |
