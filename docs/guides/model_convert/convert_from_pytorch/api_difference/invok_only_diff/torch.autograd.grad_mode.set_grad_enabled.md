## [ 仅 API 调用方式不一致 ]torch.autograd.grad_mode.set_grad_enabled

### [torch.autograd.grad\_mode.set\_grad\_enabled](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_grad_enabled.html#torch.autograd.grad_mode.set_grad_enabled)

```python
torch.autograd.grad_mode.set_grad_enabled(mode)
```

### [paddle.set\_grad\_enabled](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/set_grad_enabled_cn.html#paddle.set_grad_enabled)

```python
paddle.set_grad_enabled(mode)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
with torch.autograd.grad_mode.set_grad_enabled(is_train):
    y = x * 2

# Paddle 写法
with paddle.set_grad_enabled(is_train):
    y = x * 2
```
