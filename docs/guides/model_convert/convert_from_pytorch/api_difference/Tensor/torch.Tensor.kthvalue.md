## [ 仅参数名不一致 ]torch.Tensor.kthvalue

### [torch.Tensor.kthvalue](https://pytorch.org/docs/stable/generated/torch.Tensor.kthvalue.html)

```python
torch.Tensor.kthvalue(k, dim=None, keepdim=False)
```

### [paddle.Tensor.kthvalue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#kthvalue-k-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.kthvalue(k, axis=None, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| k       | k            | 需要沿轴查找的第 k 小，所对应的 k 值。 |
| dim     | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。 |
| keepdim | keepdim      | 是否保留指定的轴。 |
