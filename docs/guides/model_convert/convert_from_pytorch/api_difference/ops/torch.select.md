## [ 组合替代实现 ]torch.select

### [torch.select](https://pytorch.org/docs/stable/generated/torch.select.html#torch.select)

```python
torch.select(input, dim, index)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.select(a, dim=dim, index=index)

# Paddle 写法
y = paddle.index_select(a, index=paddle.to_tensor([index]), axis=dim).squeeze(dim)
```
