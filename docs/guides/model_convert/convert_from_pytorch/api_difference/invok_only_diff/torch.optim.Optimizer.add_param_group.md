## [ 仅 API 调用方式不一致 ]torch.optim.Optimizer.add_param_group

### [torch.optim.Optimizer.add\_param\_group](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.add_param_group.html#torch.optim.Optimizer.add_param_group)

```python
torch.optim.Optimizer.add_param_group(param_group)
```

### [paddle.optimizer.Optimizer._add_param_group](https://github.com/PaddlePaddle/Paddle/blob/7763841cacc6f3235e97c0cacd0a7381860e044c/python/paddle/optimizer/optimizer.py#L2093-L2136)

```python
paddle.optimizer.Optimizer._add_param_group(param_group)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
optimizer = torch.optim.SGD(pg1, lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer.add_param_group({
    'params': pg2,
    'lr': 0.1 * 2,
    'weight_decay': 0.0
})

# Paddle 写法
optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=pg1, weight_decay=0.0005)
optimizer._add_param_group({
    'params': pg2,
    'learning_rate': 0.1 * 2,
    'weight_decay': 0.0
})

```
