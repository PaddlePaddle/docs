## [ 仅参数名不一致 ]torch.Tensor.argmin

### [torch.Tensor.argmin](https://pytorch.org/docs/stable/generated/torch.Tensor.argmin.html)

```python
torch.Tensor.argmin(dim=None, keepdim=False)
```

### [paddle.Tensor.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#argmin-axis-none-keepdim-false-dtype-int64-name-none)

```python
paddle.Tensor.argmin(axis=None, keepdim=False, dtype=int64, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ------------------                 |
| dim     | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。  |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
