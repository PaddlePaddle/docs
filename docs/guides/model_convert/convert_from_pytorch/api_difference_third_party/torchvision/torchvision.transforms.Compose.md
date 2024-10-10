## [参数完全一致]torchvision.transforms.Compose

### [torchvision.transforms.Compose](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html)

```python
torchvision.transforms.Compose(
    transforms: List[Transform]
)
```

### [paddle.vision.transforms.Compose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Compose_cn.html)

```python
paddle.vision.transforms.Compose(
    transforms: Union[List[Transform], Tuple[Transform, ...]]
)

```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision                 | PaddlePaddle                | 备注                                     |
| ---------------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| transforms          | transforms                       | 用于组合的数据预处理接口实例列表。 |
