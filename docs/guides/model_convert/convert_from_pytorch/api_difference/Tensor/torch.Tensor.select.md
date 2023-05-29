## [ 组合替代实现 ]torch.Tensor.select

### [torch.Tensor.select](https://pytorch.org/docs/stable/generated/torch.Tensor.select.html?highlight=select#torch.Tensor.select)

```python
torch.Tensor.select(dim, index)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.select(dim=dim, index=index)

# Paddle 写法
y = paddle.index_select(a, index=paddle.to_tensor([index]), axis=dim).squeeze(dim)
```
