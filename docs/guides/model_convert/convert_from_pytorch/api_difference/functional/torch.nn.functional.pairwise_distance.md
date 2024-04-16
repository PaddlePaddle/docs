## [ 仅参数名不一致 ]torch.nn.functional.pairwise_distance

### [torch.nn.functional.pairwise\_distance](https://pytorch.org/docs/stable/generated/torch.nn.functional.pairwise_distance.html)

```python
torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False)
```

### [paddle.nn.functional.pairwise\_distance](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/pairwise_distance_cn.html#pairwise-distance)

```python
paddle.nn.functional.pairwise_distance(x, y, p=2., epsilon=1e-6, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| x1      | x            | 输入 Tensor，仅参数名不一致。 |
| x2      | y            | 输入 Tensor，仅参数名不一致。 |
| p       | p            | 指定 p 阶的范数。 |
| eps     | epsilon      | 添加到分母的一个很小值，避免发生除零错误。仅参数名不一致。 |
| keepdim | keepdim      | 是否保留输出 Tensor 减少的维度。 |
