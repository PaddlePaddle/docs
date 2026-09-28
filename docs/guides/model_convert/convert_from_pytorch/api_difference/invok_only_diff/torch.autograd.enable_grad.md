## [ 仅 API 调用方式不一致 ]torch.autograd.enable_grad

### [torch.autograd.enable_grad](https://docs.pytorch.org/docs/stable/generated/torch.enable_grad.html#torch.enable_grad)

```python
torch.autograd.enable_grad(*args, **kwargs)
```

### [paddle.autograd.enable\_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/enable_grad_cn.html#paddle.autograd.enable_grad)

```python
paddle.autograd.enable_grad(*args, **kwargs)
```

两者功能一致，Paddle 也支持在 ``paddle.autograd`` 命名空间下调用，具体如下：

### 转写示例

```python
# PyTorch 写法
@torch.autograd.enable_grad()
def doubler(x):
    return x * 2

with torch.no_grad():
    result = doubler(x)

# Paddle 写法
@paddle.autograd.enable_grad()
def doubler(x):
    return x * 2

with paddle.no_grad():
    result = doubler(x)

```
