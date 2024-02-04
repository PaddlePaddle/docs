## [ 仅参数名不一致 ]torch.Tensor.mode

### [torch.Tensor.mode](https://pytorch.org/docs/stable/generated/torch.Tensor.mode.html)

```python
torch.Tensor.mode(dim=None, keepdim=False)
```

### [paddle.Tensor.mode](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#mode-axis-1-keepdim-false-name-none)

```python
paddle.Tensor.mode(axis=-1, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| dim     | axis         | 指定对输入 Tensor 进行运算的轴。 |
| keepdim | keepdim      | 是否保留指定的轴。 |
