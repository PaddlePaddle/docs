## [参数不一致]torch.Tensor.repeat

### [torch.Tensor.repeat](https://pytorch.org/docs/1.13/generated/torch.Tensor.repeat.html)

```python
    torch.Tensor.repeat(*sizes)
```

### [paddle.Tensor.tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#tile-repeat-times-name-none)

```python
    paddle.Tensor.tile(repeat_times, name=None)
```

### 参数映射
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *sizes | repeat_times | *sizes ：参数类型为 torch.Size 或 int... ; repeat_times ：参数类型为 list 、 tuple 或 Tensor。 torch 参数为可变参数时需要转写。|

### 转写示例
#### *sizes: 各个维度重复的次数，可变参数用法
```python
    # pytorch
    x = torch.randn(2, 3, 5)
    x_repeat = x.repeat(4,2,1)

    # paddle
    x = paddle.randn([2, 3, 5])
    x_tile_tuple = x.tile((4,2,1))
```
