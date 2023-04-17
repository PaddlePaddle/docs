## [ 组合替代实现 ]torch.nn.functional.upsample_bilinear

### [torch.nn.functional.upsample_bilinear](https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample_bilinear.html#torch.nn.functional.upsample_bilinear)

```python
torch.nn.functional.upsample_bilinear(input, size=None, scale_factor=None)
```

### 功能介绍
Paddle 无此 API，需要组合实现。

```python
# PyTorch 写法
torch.nn.functional.upsample_bilinear(input=input, scale_factor=2)

# Paddle 写法
paddle.nn.functional.upsample(input=input, scale_factor=2, mode='bilinear', align_corners=True)
```
