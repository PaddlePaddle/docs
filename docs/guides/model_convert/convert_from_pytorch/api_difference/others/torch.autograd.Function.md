## [torch 参数更多]torch.autograd.Function

### [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)

```python
torch.autograd.Function(*args, **kwargs)
```

### [paddle.autograd.PyLayer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayer_cn.html#paddle.autograd.PyLayer)

```python
paddle.autograd.PyLayer()
```

其中 PyTorch 相比 Paddle 支持更多其他参数，但该类一般用于继承实现，不会调用其参数。

### 参数映射

| PyTorch | PaddlePaddle | 备注                                      |
| ------- | ------------ | ----------------------------------------- |
| args    | -            | 自定义参数，Paddle 无此参数，可直接删除。 |
| kwargs  | -            | 自定义参数，Paddle 无此参数，可直接删除。 |
