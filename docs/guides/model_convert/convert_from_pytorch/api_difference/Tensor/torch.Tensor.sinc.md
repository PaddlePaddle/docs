## [ 组合替代实现 ]torch.Tensor.sinc

### [torch.Tensor.sinc](https://pytorch.org/docs/stable/generated/torch.Tensor.sinc.html#torch.Tensor.sinc)

```python
torch.Tensor.sinc()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = a.sinc()

# Paddle 写法
import numpy
y = paddle.where(a==0, x=paddle.to_tensor([1], dtype=a.dtype), y=paddle.sin(numpy.pi*a)/(numpy.pi*a))
```
