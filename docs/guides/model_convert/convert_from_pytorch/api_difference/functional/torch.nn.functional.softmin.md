## [ 组合替代实现 ]torch.nn.functional.softmin

### [torch.nn.functional.softmin](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmin.html#torch.nn.functional.softmin)

```python
torch.nn.functional.softmin(input, dim=None, _stacklevel=3, dtype=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.nn.functional.softmin(input, dim=1)

# Paddle 写法
paddle.nn.functinal.softmax(-input, axis=1)
```
