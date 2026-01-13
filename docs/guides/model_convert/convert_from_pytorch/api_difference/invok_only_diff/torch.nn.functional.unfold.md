## [ 仅 API 调用方式不一致 ]torch.nn.functional.unfold

### [torch.nn.functional.unfold](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html)

```python
torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
```

### [paddle.compat.nn.functional.unfold](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/transformer.py#L29)
```python
paddle.compat.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
out = torch.nn.functional.unfold(input, kernel_size)

# Paddle 写法
out = paddle.compat.nn.functional.unfold(input, kernel_size)
```
