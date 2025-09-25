## [torch 参数更多 ]torch.polar

### [torch.polar](https://pytorch.org/docs/stable/generated/torch.polar.html#torch.polar)
```python
torch.polar(abs, angle, *, out=None)
```

### [paddle.polar](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/polar_cn.html)

```python
paddle.polar(abs, angle, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                           |
| ------- | ------------ | ---------------------------------------------- |
| abs     | abs          | 输入的模，参数完全一致。           |
| angle   | angle        | 输入的相位角，参数完全一致。           |
| out     | -            | 输出 Tensor，Paddle 无此参数，需要转写。 |


###  转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.polar(abs, angle, out=y)

# Paddle 写法
paddle.assign(paddle.polar(abs, angle), y)
```
