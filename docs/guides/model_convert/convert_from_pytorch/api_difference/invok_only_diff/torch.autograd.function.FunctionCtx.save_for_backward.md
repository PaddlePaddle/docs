## [ 仅 API 调用方式不一致 ]torch.autograd.function.FunctionCtx.save_for_backward

### [torch.autograd.function.FunctionCtx.save\_for\_backward](https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html#torch.autograd.function.FunctionCtx.save_for_backward)

```python
torch.autograd.function.FunctionCtx.save_for_backward(*tensors)
```

### [paddle.autograd.PyLayerContext.save_for_backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#save-for-backward-tensors)

```python
paddle.autograd.PyLayerContext.save_for_backward(*tensors)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class cus_tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.nn.functional.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad

# Paddle 写法
class cus_tanh(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.nn.functional.tanh(x=x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad
```
