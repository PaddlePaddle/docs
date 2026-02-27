## [ 仅 API 调用方式不一致 ]torch.nn.functional.unfold

### [torch.nn.functional.unfold](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html#torch.nn.functional.unfold)

```python
torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
```

### [paddle.compat.nn.functional.unfold](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/transformer.py#L29)
```python
paddle.compat.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
out = torch.nn.functional.unfold(input, kernel_size)

# Paddle 写法
out = paddle.compat.nn.functional.unfold(input, kernel_size)
```
