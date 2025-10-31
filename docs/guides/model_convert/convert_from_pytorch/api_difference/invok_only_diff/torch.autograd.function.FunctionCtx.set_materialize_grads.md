## [ 仅 API 调用方式不一致 ]torch.autograd.function.FunctionCtx.set_materialize_grads

### [torch.autograd.function.FunctionCtx.set_materialize_grads](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.html#torch.autograd.function.FunctionCtx.set_materialize_grads)

```python
torch.autograd.function.FunctionCtx.set_materialize_grads(value)
```

### [paddle.autograd.PyLayerContext.set_materialize_grads](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayerContext/set_materialize_grads_cn.html#paddle/autograd/PyLayerContext/set_materialize_grads_cn#cn-api-paddle-autograd-PyLayerContext-set_materialize_grads)

```python
paddle.autograd.PyLayerContext.set_materialize_grads(value)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class cus_tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x + x + x, x + x

    @staticmethod
    def backward(ctx, grad, grad2):
        assert grad2 == None
        return grad

# Paddle 写法
class cus_tanh(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x + x + x, x + x

    @staticmethod
    def backward(ctx, grad, grad2):
        assert grad2 == None
        return grad
```
