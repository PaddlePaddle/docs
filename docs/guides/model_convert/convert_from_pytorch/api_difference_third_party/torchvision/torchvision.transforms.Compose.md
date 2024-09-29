## [参数完全一致]torchvision.transforms.Compose

### [torchvision.transforms.Compose](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html)

```python
torchvision.transforms.Compose(transforms: list of Transform objects)
```

### [paddle.vision.transforms.Compose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Compose_cn.html)

```python
paddle.vision.transforms.Compose(transforms: list | tuple)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision                 | paddle                | 备注                                     |
| ---------------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| transforms (list of Transform objects)         | transforms (list \| tuple)                      | torch 仅支持 list，paddle 支持 list 或 tuple |
