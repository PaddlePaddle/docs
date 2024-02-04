## [ 仅参数名不一致 ]torch.Tensor.amax

### [torch.Tensor.amax](https://pytorch.org/docs/stable/generated/torch.Tensor.amax.html)

```python
torch.Tensor.amax(dim=None, keepdim=False)
```

### [paddle.Tensor.amax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#amax-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.amax(axis=None, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ------------------                 |
| dim     | axis         | 求最大值运算的维度。                 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
