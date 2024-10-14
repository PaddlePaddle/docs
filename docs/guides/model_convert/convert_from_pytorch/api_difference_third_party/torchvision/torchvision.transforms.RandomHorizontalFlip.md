## [paddle 参数更多]torchvision.transforms.RandomHorizontalFlip

### [torchvision.transforms.RandomHorizontalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html?highlight=randomhorizontalflip#torchvision.transforms.RandomHorizontalFlip)

```python
torchvision.transforms.RandomHorizontalFlip(p: float = 0.5)
```

### [paddle.vision.transforms.RandomHorizontalFlip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomHorizontalFlip_cn.html)

```python
paddle.vision.transforms.RandomHorizontalFlip(
    prob: float = 0.5,
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                    |
| ------------- | ------------- | --------------------- |
| p             | prob          | 翻转概率，仅参数名不一致。        |
| -             | keys          | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。     |
