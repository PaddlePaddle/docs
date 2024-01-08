## [ 组合替代实现 ]torch.Tensor.aminmax

### [torch.Tensor.aminmax](https://pytorch.org/docs/stable/generated/torch.Tensor.aminmax.html#torch.Tensor.aminmax)

```python
torch.Tensor.aminmax(*, dim=None, keepdim=False)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
input.aminmax(dim=-1, keepdim=True)

# Paddle 写法
y = tuple([paddle.amin(input, axis=-1, keepdim=True), paddle.amax(input, axis=-1, keepdim=True)])
```
