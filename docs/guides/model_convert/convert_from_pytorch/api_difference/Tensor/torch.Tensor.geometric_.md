## [ torch 参数更多 ]torch.Tensor.geometric_

### [torch.Tensor.geometric_](https://pytorch.org/docs/stable/generated/torch.Tensor.geometric_.html)

```python
torch.Tensor.geometric_(p, *, generator=None)
```

### [paddle.Tensor.geometric_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html)

```python
paddle.Tensor.geometric_(probs, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| p   | probs            | 输入 Tensor，仅参数名不一致。 |
| generator | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
