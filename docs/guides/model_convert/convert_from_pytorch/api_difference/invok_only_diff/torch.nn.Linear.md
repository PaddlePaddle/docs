## [ 仅 API 调用方式不一致 ]torch.nn.Linear

### [torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

```python
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

### [paddle.compat.nn.Linear](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/nn/__init__.py#L470)

```python
paddle.compat.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
layer = torch.nn.Linear(in_features=16, out_features=16)

# Paddle 写法
layer = paddle.compat.nn.Linear(in_features=16, out_features=16)
```
