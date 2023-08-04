## [参数完全一致]torch.autograd.function.FunctionCtx.set_materialize_grads

### [torch.autograd.function.FunctionCtx.set_materialize_grads](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html#torch.autograd.function.FunctionCtx.set_materialize_grads)

```python
torch.autograd.function.FunctionCtx.set_materialize_grads(value)
```

### [paddle.autograd.PyLayerContext.set_materialize_grads](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/autograd/PyLayerContext_cn.html#set-materialize-grads-self-value)

```python
paddle.autograd.PyLayerContext.set_materialize_grads(value)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                   |
| ------- | ------------ | -------------------------------------- |
| value   | value        | 是否要框架来初始化未初始化的反向梯度。 |
