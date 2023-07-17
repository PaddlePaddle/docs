## [ 仅参数名不一致 ]torch.Tensor.remainder
### [torch.Tensor.roll](https://pytorch.org/docs/stable/generated/torch.Tensor.remainder.html?highlight=torch+tensor+remainder#torch.Tensor.remainder)

```python
torch.Tensor.remainder(divisor)
```

### [paddle.remainder](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/remainder_cn.html#remainder)

```python
paddle.remainder(x, y, name=None)
```


其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| divisor         | y            | 除数，Pytorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor, 仅参数名不一致。   |
