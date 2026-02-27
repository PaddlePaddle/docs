## [ 输入参数用法不一致 ]torch.optim.Optimizer.zero_grad
### [torch.optim.Optimizer.zero\_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad)
```python
torch.optim.Optimizer.zero_grad(set_to_none=True)
```

### [paddle.optimizer.Optimizer.clear_gradients](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Optimizer_cn.html#clear-grad)
```python
paddle.optimizer.Optimizer.clear_gradients(set_to_zero=True)
```

PyTorch 的 `Optimizer.zero_grad` 参数与 Paddle 的 `Optimizer.clear_gradients` 参数用法刚好相反，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                             |
| ----------- | ------------ | ------------------------------------------------ |
| set_to_none | set_to_zero  | 设置如何清空梯度，PyTorch 默认 set_to_none 为 True，Paddle 默认 set_to_zero 为 True，两者功能刚好相反，Paddle 需设置为 False。需要转写。 |

### 转写示例
#### set_to_none 转写为 set_to_zero
```python
# PyTorch 写法
torch.optim.Optimizer.zero_grad(set_to_none=True)

# Paddle 写法
paddle.optimizer.Optimizer.clear_gradients(set_to_zero=False)
```
