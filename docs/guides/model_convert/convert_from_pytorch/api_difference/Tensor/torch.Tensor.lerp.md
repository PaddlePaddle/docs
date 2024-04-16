## [ 仅参数名不一致 ]torch.Tensor.lerp

### [torch.Tensor.lerp](https://pytorch.org/docs/stable/generated/torch.Tensor.lerp.html)

```python
torch.Tensor.lerp(end, weight)
```

### [paddle.Tensor.lerp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#lerp-x-y-weight-name-none)

```python
paddle.Tensor.lerp(y, weight, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| end     | y            | 输入的 Tensor，作为线性插值结束的点，仅参数名不一致。 |
| weight  | weight       | 给定的权重值。 |
