## [ 组合替代实现 ]torch.Tensor.msort

### [torch.Tensor.msort](https://pytorch.org/docs/stable/generated/torch.Tensor.msort.html#torch.Tensor.msort)

```python
torch.Tensor.msort()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.msort()

# Paddle 写法
y = paddle.sort(a, axis=0)
```
