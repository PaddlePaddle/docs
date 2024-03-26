## [torch 参数更多]torch.nn.functional.poisson_nll_loss

### [torch.nn.functional.poisson_nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html?highlight=gumbel_softmax#torch.nn.functional.gumbel_softmax)

```python
torch.nn.functional.poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```

### [paddle.nn.functional.poisson_nll_loss](https://www.paddlepaddle.org.cn/documentation/api/paddle/nn/functional/poisson_nll_loss_cn.html)

```python
paddle.nn.functional.poisson_nll_loss(input, label, log_input=False, full=False, eps=1e-8, reduction='mean', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                           |
|--------|--------------|------------------------------|
| input | input           | 参数完全相同                         |
| target    | label  | 仅参数名不同                        |
| log_input   | log_input             | 参数完全相同 |
| full    |  full | 参数完全相同                       |
| size_average    |axis| 仅参数名不同                         |
| eps    |eps| 参数完全相同                          |
| reduce    || Paddle无此参数，可直接删除                         |
| reduction    |reduction| 参数完全相同                         |
