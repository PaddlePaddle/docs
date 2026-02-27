## [ 仅 API 调用方式不一致 ]torch.autograd.function.FunctionCtx.mark_non_differentiable

### [torch.autograd.function.FunctionCtx.mark\_non\_differentiable](https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html#torch.autograd.function.FunctionCtx.mark_non_differentiable)

```python
torch.autograd.function.FunctionCtx.mark_non_differentiable(*args)
```

### [paddle.autograd.PyLayerContext.mark_non_differentiable](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#mark-non-differentiable-self-tensors)

```python
paddle.autograd.PyLayerContext.mark_non_differentiable(*args)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class cus_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        a = x + x
        b = x + x + x
        ctx.mark_non_differentiable(a)
        return a, b

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        grad_x = 3 * grad_b
        return grad_x

# Paddle 写法
class cus_func(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        a = x + x
        b = x + x + x
        ctx.mark_non_differentiable(a)
        return a, b

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        grad_x = 3 * grad_b
        return grad_x
```
