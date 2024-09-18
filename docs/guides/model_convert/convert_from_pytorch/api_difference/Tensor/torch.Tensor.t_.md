## [ 组合替代实现 ] torch.Tensor.t_

### [torch.Tensor.t_](https://pytorch.org/docs/stable/generated/torch.Tensor.t_.html#torch.Tensor.t_)

```python
torch.Tensor.t_()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
x.t_()

# Paddle 写法
paddle.assign(x.t(), x)
```
