## [ 仅 API 调用方式不一致 ]torch.nn.Softmax

### [torch.nn.Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax)

```python
torch.nn.Softmax(dim=None)
```

### [paddle.compat.nn.Softmax](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/__init__.py#L613)

```python
paddle.compat.nn.Softmax(dim=None)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
layer = torch.nn.Softmax()

# Paddle 写法
layer = paddle.compat.nn.Softmax()
```
