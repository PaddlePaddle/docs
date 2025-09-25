## [仅参数名不一致]torch.autograd.function.FunctionCtx.mark_non_differentiable

### [torch.autograd.function.FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html#torch.autograd.function.FunctionCtx.mark_non_differentiable)

```python
torch.autograd.function.FunctionCtx.mark_non_differentiable(*args)
```

### [paddle.autograd.PyLayerContext.mark_non_differentiable](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#mark-non-differentiable-self-tensors)

```python
paddle.autograd.PyLayerContext.mark_non_differentiable(*tensors)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| args    | tensors      | 需要标记不需要反向的 Tensor。 |
