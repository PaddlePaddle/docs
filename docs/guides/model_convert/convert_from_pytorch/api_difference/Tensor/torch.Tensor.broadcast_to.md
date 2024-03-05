## [ 仅参数名不一致 ]torch.Tensor.broadcast_to

### [torch.Tensor.broadcast\_to](https://pytorch.org/docs/stable/generated/torch.Tensor.broadcast_to.html)

```python
torch.Tensor.broadcast_to(size)
```

### [paddle.Tensor.broadcast\_to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#broadcast-to-shape-name-none)

```python
paddle.Tensor.broadcast_to(shape, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| size    | shape        | 给定输入 x 扩展后的形状，仅参数名不一致。  |
