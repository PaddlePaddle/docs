## [ 仅 API 调用方式不一致 ]torch.nn.functional.linear

### [torch.nn.functional.linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html)

```python
torch.nn.functional.linear(input, weight, bias=None)
```

### [paddle.compat.nn.functional.linear](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/functional/__init__.py#L204)
```python
paddle.compat.nn.functional.linear(input, weight, bias=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
out = torch.nn.functional.linear(input, weight, bias)

# Paddle 写法
out = paddle.compat.nn.functional.linear(input, weight, bias)
```
