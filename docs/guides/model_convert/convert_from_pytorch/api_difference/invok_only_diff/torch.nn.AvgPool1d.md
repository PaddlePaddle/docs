## [ 仅 API 调用方式不一致 ]torch.nn.AvgPool1d

### [torch.nn.AvgPool1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d)

```python
torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

### [paddle.compat.nn.AvgPool1d](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/__init__.py#L67)

```python
paddle.compat.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
layer = torch.nn.AvgPool1d(2)

# Paddle 写法
layer = paddle.compat.nn.AvgPool1d(2)
```
