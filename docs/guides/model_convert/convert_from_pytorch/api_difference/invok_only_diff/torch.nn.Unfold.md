## [ 仅 API 调用方式不一致 ]torch.nn.Unfold

### [torch.nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold)

```python
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
```

### [paddle.compat.nn.Unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/nn/Unfold_cn.html#paddle/compat/nn/Unfold_cn#cn-api-paddle-compat-nn-Unfold)

```python
paddle.compat.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
unfold = torch.nn.Unfold(kernel_size=(2, 2))

# Paddle 写法
unfold = paddle.compat.nn.Unfold(kernel_size=(2, 2))

```
