## [ torch 参数更多 ] torch.Tensor.random_

### [torch.Tensor.random_](https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html)

```python
torch.Tensor.random_(from=0, to=None, *, generator=None)
```

### [paddle.Tensor.uniform_](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/Tensor/uniform__en.html)

```python
paddle.Tensor.uniform_(min: float = - 1.0, max: float = 1.0, seed: int = 0)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下:

### 参数映射


| PyTorch   | PaddlePaddle | 备注                                                                                 |
| --------- | ------------ | ------------------------------------------------------------------------------------ |
| from      | min          | 随机数生成范围的起始值，仅参数名不一致。                                             |
| to        | max          | 随机数生成范围的结束值，仅参数名不一致。                                             |
| generator | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -         | seed         | 随机数种子。PyTorch 无此参数，Paddle 保持默认即可。                                  |
