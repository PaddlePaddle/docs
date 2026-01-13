## [ 仅 API 调用方式不一致 ]torch.nn.AvgPool2d

### [torch.nn.AvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)

```python
torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

### [paddle.compat.nn.AvgPool2d](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/__init__.py#L171)

```python
paddle.compat.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
layer = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

# Paddle 写法
layer = paddle.compat.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
```
