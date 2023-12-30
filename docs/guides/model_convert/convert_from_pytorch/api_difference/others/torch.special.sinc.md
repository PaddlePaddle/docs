## [ 组合替代实现 ]torch.special.sinc

### [torch.special.sinc](https://pytorch.org/docs/stable/special.html#torch.special.sinc)

```python
torch.special.sinc(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.special.sinc(a)

# Paddle 写法
import numpy
y = paddle.where(a==0, x=paddle.to_tensor([1], dtype=a.dtype), y=paddle.sin(numpy.pi*a)/(numpy.pi*a))
```
