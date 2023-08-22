## [参数完全一致]torch.autograd.function.FunctionCtx.save_for_backward

### [torch.autograd.function.FunctionCtx.save_for_backward](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html#torch.autograd.function.FunctionCtx.save_for_backward)

```python
torch.autograd.function.FunctionCtx.save_for_backward(*tensors)
```

### [paddle.autograd.PyLayerContext.save_for_backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#save-for-backward-tensors)

```python
paddle.autograd.PyLayerContext.save_for_backward(*tensors)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                  |
| ------- | ------------ | --------------------- |
| tensors | tensors      | 需要被暂存的 Tensor。 |
