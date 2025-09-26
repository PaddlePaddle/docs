## [ 仅参数名不一致 ]torch.Tensor.amin

### [torch.Tensor.amin](https://pytorch.org/docs/stable/generated/torch.Tensor.amin.html)

```python
torch.Tensor.amin(dim=None, keepdim=False)
```

### [paddle.Tensor.amin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#amin-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.amin(axis=None, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ------------------                 |
| dim     | axis         | 求最小值运算的维度，仅参数名不一致。  |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
