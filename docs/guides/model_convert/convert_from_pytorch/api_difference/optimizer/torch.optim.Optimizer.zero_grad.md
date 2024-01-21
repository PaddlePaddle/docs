## [torch 参数更多]torch.optim.Optimizer.zero_grad

### [torch.optim.Optimizer.zero_grad](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.optim.Optimizer.zero_grad)

```python
torch.optim.Optimizer.zero_grad(set_to_none=True)
```

### [paddle.optimizer.Optimizer.clear_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Optimizer_cn.html#clear-grad)

```python
paddle.optimizer.Optimizer.clear_grad()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                             |
| ----------- | ------------ | ------------------------------------------------ |
| set_to_none | -            | 是否设置为 None，Paddle 无此参数，暂无转写方式。 |
