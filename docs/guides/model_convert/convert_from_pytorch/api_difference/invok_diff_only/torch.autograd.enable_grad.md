## [ 仅 API 调用方式不一致 ]torch.autograd.enable_grad

### [torch.autograd.enable_grad](https://pytorch.org/docs/stable/generated/torch.autograd.html#torch.autograd.enable_grad)

```python
torch.autograd.enable_grad(*args, **kwargs)
```

### [paddle.enable_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/enable_grad_cn.html#paddle/enable_grad_cn#cn-api-paddle-enable_grad)

```python
paddle.enable_grad(*args, **kwargs)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
@torch.autograd.enable_grad()
def doubler(x):
    return x * 2

with torch.no_grad():
    result = doubler(x)

# Paddle 写法
@paddle.enable_grad()
def doubler(x):
    return x * 2

with paddle.no_grad():
    result = doubler(x)

```
