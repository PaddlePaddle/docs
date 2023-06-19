## [参数不一致]torch.optim.Optimizer.zero_grad

### [torch.optim.Optimizer.zero_grad](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html?highlight=zero_grad#torch.optim.Optimizer.zero_grad)

```
torch.optim.Optimizer.zero_grad(set_to_none=True)
```

### [paddle.optimizer.Optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Optimizer_cn.html)

```
paddle.optimizer.Optimizer.clear_grad(set_to_zero=True)
```

两者参数不一致，torch 和 paddle 默认设置相反。torch 默认将模型权重的梯度设置为 None ，paddle 则默认设置成 0 。具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                   |
| ----------- | ------------ | ------------------------------------------------------ |
| set_to_none | set_to_zero  | 表示优化器梯度清零后模型参数的值被设置成 0 还是 None。 |
