## [ torch 参数更多 ]torch.Tensor.cauchy_

### [torch.Tensor.cauchy_](https://pytorch.org/docs/stable/generated/torch.Tensor.cauchy_.html)

```python
torch.Tensor.cauchy_(median=0, sigma=1, *, generator=None)
```

### [paddle.Tensor.cauchy_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html)

```python
paddle.Tensor.cauchy_(loc=0, scale=1, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                 |
| --------- | ------------ | ------------------------------------------------------------------------------------ |
| median    | loc          | 输入 Tensor，仅参数名不一致。                                                        |
| sigma     | scale        | 输入 Tensor，仅参数名不一致。                                                        |
| *         | -            | 其他参数                                                                             |
| generator | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
