## [ 组合替代实现 ]torch.aminmax

### [torch.aminmax](https://pytorch.org/docs/stable/generated/torch.aminmax.html#torch.aminmax)

```python
torch.aminmax(input, *, dim=None, keepdim=False, out=None)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.aminmax(input, dim=-1, keepdim=True)

# Paddle 写法
y = tuple([paddle.amin(input, axis=-1, keepdim=True), paddle.amax(input, axis=-1, keepdim=True)])
```
