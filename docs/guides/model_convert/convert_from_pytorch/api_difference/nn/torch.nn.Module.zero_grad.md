## [参数不一致]torch.nn.Module.zero_grad

### [torch.nn.Module.zero_grad](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)

```python
torch.nn.Module.zero_grad(set_to_none=True)
```

### [paddle.nn.Layer.clear_gradients](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#clear-gradients)

```python
paddle.nn.Layer.clear_gradients(set_to_zero=True)
```

PyTorch 的 `Module.zero_grad` 参数与 Paddle 的 `Layer.clear_gradients` 参数用法不同，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                             |
| ----------- | ------------ | ------------------------------------------------ |
| set_to_none | set_to_zero  | 设置如何清空梯度，PyTorch 默认设置为 None，Paddle 默认设置为 0，需要转写。 |

### 转写示例

```python
# PyTorch 写法
torch.nn.Module.zero_grad(set_to_none=True)

# Paddle 写法
paddle.nn.Layer.clear_gradients(set_to_zero=False)
```
