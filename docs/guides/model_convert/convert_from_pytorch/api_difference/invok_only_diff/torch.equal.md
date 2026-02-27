## [ 仅 API 调用方式不一致 ]torch.equal

### [torch.equal](https://docs.pytorch.org/docs/stable/generated/torch.equal.html#torch.equal)

```python
torch.equal(input, other)
```

### [paddle.compat.equal](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/compat/__init__.py#L70)

```python
paddle.compat.equal(input, other)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.equal(a, b)

# Paddle 写法
result = paddle.compat.equal(a, b)
```
