## [ 组合替代实现 ]torch.sinc

### [torch.sinc](https://pytorch.org/docs/stable/generated/torch.sinc.html#torch.sinc)

```python
torch.sinc(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.sinc(a)

# Paddle 写法
import numpy
y = paddle.where(a==0, x=paddle.to_tensor([1], dtype=a.dtype), y=paddle.sin(numpy.pi*a)/(numpy.pi*a))
```
