## [ 组合替代实现 ]torch.Tensor.xlogy_

### [torch.Tensor.xlogy_](https://pytorch.org/docs/stable/generated/torch.Tensor.xlogy_.html#torch.Tensor.xlogy_)

```python
torch.Tensor.xlogy_(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
a.xlogy_(b)

# Paddle 写法
paddle.assign(a * paddle.log(b), a)
```
