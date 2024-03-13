## [ 仅参数默认值不一致 ]torch.rrelu

### [torch.rrelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.rrelu.html#torch.nn.functional.rrelu)

```python
torch.rrelu(input, lower=1./8, upper=1./3, training=False)
```

### [paddle.nn.functional.rrelu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/rrelu_cn.html)

```python
paddle.nn.functional.rrelu(x, lower=1./8, upper=1./3, training=True, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名与参数默认值不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                                                                            |
| -------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input    | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| lower    | lower        | 负值斜率的随机值范围下限。                                                                                      |
| upper    | upper        | 负值斜率的随机值范围上限。                                                                                      |
| training | training     | 标记是否为训练阶段，仅参数默认值不一致。                                                                                            |
