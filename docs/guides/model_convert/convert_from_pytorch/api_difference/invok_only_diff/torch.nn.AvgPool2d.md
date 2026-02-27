## [ 仅 API 调用方式不一致 ]torch.nn.AvgPool2d

### [torch.nn.AvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d)

```python
torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

### [paddle.compat.nn.AvgPool2d](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/__init__.py#L171)

```python
paddle.compat.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
layer = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

# Paddle 写法
layer = paddle.compat.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
```
