## [ 组合替代实现 ]torch.nn.functional.upsample_nearest

### [torch.nn.functional.upsample_nearest](https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample_nearest.html#torch.nn.functional.upsample_nearest)

```python
torch.nn.functional.upsample_nearest(input, size=None, scale_factor=None)
```

### 功能介绍
Paddle 无此 API，需要组合实现。

```python
# PyTorch 写法
torch.nn.functional.upsample_nearest(input=input, scale_factor=2)

# Paddle 写法
paddle.nn.functional.upsample(input=input, scale_factor=2, mode='nearest')
```
