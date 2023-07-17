## [ 仅参数名不一致 ]torch.Tensor.remainder_
### [torch.Tensor.roll](https://pytorch.org/docs/stable/generated/torch.Tensor.remainder_.html?highlight=torch+tensor+remainder_#torch.Tensor.remainder_)

```python
torch.Tensor.remainder_(divisor)
```

### [paddle.remainder_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/remainder__cn.html#remainder)

```python
paddle.remainder_(x, y, name=None)
```


其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| divisor         | y            | 除数，Pytorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor，仅参数名不一致。   |
