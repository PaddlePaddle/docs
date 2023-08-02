## [torch 参数更多]torch.nn.functional.rrelu

### [torch.nn.functional.rrelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.rrelu.html#torch.nn.functional.rrelu)

```python
torch.nn.functional.rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False)
```

### [paddle.nn.functional.rrelu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/rrelu_cn.html)

```python
paddle.nn.functional.rrelu(x, lower=1. / 8., upper=1. / 3., training=True, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                                                                            |
| -------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input    | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| lower    | lower        | 负值斜率的随机值范围下限。                                                                                      |
| upper    | upper        | 负值斜率的随机值范围上限。                                                                                      |
| training | training     | 标记是否为训练阶段。                                                                                            |
| inplace  | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
