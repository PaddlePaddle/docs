## [ 仅参数名不一致 ]torch.Tensor.baddbmm

### [torch.Tensor.baddbmm](https://pytorch.org/docs/stable/generated/torch.Tensor.baddbmm.html#torch.Tensor.baddbmm)

```python
torch.Tensor.baddbmm(batch1, batch2, *, beta=1, alpha=1)
```

### [paddle.Tensor.baddbmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tensor)

```python
paddle.Tensor.baddbmm(x, y, beta=1, alpha=1, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| batch1 | x | 表示输入的第一个 Tensor ，仅参数名不一致。 |
| batch2 | y | 表示输入的第二个 Tensor ，仅参数名不一致。 |
| beta | beta | 表示乘以 input 的标量。 |
| alpha | alpha | 表示乘以 batch1 * batch2 的标量。 |
