## [ 仅 API 调用方式不一致 ]torch.autograd.function.FunctionCtx.saved_tensors

### [torch.autograd.function.FunctionCtx.saved_tensors](https://github.com/pytorch/pytorch/blob/c03e1676230434e4d84d997682b2fbe892fb05f1/torch/autograd/function.py#L839-L845)

```python
torch.autograd.function.FunctionCtx.saved_tensors
```

### [paddle.autograd.PyLayerContext.saved_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext_cn.html#saved-tensor)

```python
paddle.autograd.PyLayerContext.saved_tensor()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class cus_tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.nn.functional.tanh(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, y = ctx.saved_tensors
        grad = y + dy + 1
        return grad

# Paddle 写法
class cus_tanh(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.nn.functional.tanh(x=x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, y = ctx.saved_tensor()
        grad = y + dy + 1
        return grad
```
