## [ 组合替代实现 ]torch.Tensor.xlogy

### [torch.Tensor.xlogy](https://pytorch.org/docs/stable/generated/torch.Tensor.xlogy.html#torch.Tensor.xlogy)

```python
torch.Tensor.xlogy(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.xlogy(b)

# Paddle 写法
y = a * paddle.log(b)
```
