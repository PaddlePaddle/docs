## [ 组合替代实现 ]torch.Tensor.masked_fill
### [torch.Tensor.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html?highlight=masked_fill#torch.Tensor.masked_fill)

```python
torch.Tensor.masked_fill(mask, value)
```

torch 是类成员方式，paddle 无 masked_fill 函数，需要组合实现。

### 转写示例

```python
# torch 写法
x.masked_fill(mask, value)

# paddle 写法
out = paddle.full(x.shape, value, x.dtype)
x = paddle.where(mask, out, x)
```
