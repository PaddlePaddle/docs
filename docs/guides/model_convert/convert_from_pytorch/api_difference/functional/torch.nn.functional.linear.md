## [ 仅参数名不一致 ]torch.nn.functional.linear

### [torch.nn.functional.linear](https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html?highlight=linear#torch.nn.functional.linear)

```python
torch.nn.functional.linear(input,
                           weight,
                           bias=None)
```

### [paddle.nn.functional.linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/linear_cn.html)

```python
paddle.nn.functional.linear(x,
                            weight,
                            bias=None,
                            name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor ，仅参数名不一致。                                     |
| weight          | weight         | 表示权重 Tensor 。                                     |
| bias          | bias         | 表示偏重 Tensor 。                                     |
