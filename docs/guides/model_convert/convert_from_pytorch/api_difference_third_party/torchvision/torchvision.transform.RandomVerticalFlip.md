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

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| p (float)                                   | prob (float)                                 | 翻转概率，参数名不同。                                       |
| -                                           | keys (list[str] or tuple[str] = None)        | Paddle 支持 `keys` 参数。             |
