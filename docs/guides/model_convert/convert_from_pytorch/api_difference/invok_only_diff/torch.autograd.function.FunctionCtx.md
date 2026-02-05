## [ 仅 API 调用方式不一致 ]torch.autograd.function.FunctionCtx

### [torch.autograd.function.FunctionCtx](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.html#torch.autograd.function.FunctionCtx)

```python
torch.autograd.function.FunctionCtx(*args, **kwargs)
```

### [paddle.autograd.PyLayerContext](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#paddle.autograd.PyLayerContext)

```python
paddle.autograd.PyLayerContext(*args, **kwargs)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.x
        grad_input = grad_output * 2
        return grad_input

# Paddle 写法
class MyFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.x
        grad_input = grad_output * 2
        return grad_input
```
