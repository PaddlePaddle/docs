## [ 仅参数名不一致 ]torch.Tensor.unfold

### [torch.Tensor.unfold](https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html?highlight=unfold#torch.Tensor.unfold)

```python
torch.Tensor.unfold(dimension,
                size,
                step)
```

### [paddle.Tensor.unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#unfold-x-axis-size-step-name-none)

```python
paddle.Tensor.unfold(axis,
                size,
                step,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
| ------- | ------------ | ------------------------------------------------------------ |
| dimension    |     axis      |           表示需要提取的维度，仅参数名不一致。           |
| size      |     size      |           表示需要提取的窗口长度。           |
| step       |     step     | 表示每次提取跳跃的步长。 |
