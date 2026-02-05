## [ 仅 API 调用方式不一致 ]torch.autograd.Function.forward

### [torch.autograd.Function.forward](https://docs.pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward)

```python
torch.autograd.Function.forward(*args, **kwargs)
```

### [paddle.autograd.PyLayer.forward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayer/forward_cn.html#paddle/autograd/PyLayer/forward_cn#cn-api-paddle-autograd-PyLayer-forward)

```python
paddle.autograd.PyLayer.forward(ctx, *args, **kwargs)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class cus_tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, func=torch.square):
        ctx.func = func
        y = func(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad

# Paddle 写法
class cus_tanh(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, func=paddle.square):
        ctx.func = func
        y = func(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad
```
