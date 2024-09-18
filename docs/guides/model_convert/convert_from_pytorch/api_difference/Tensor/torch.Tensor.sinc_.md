## [ 组合替代实现 ]torch.Tensor.sinc_

### [torch.Tensor.sinc_](https://pytorch.org/docs/stable/generated/torch.Tensor.sinc_.html#torch.Tensor.sinc_)

```python
torch.Tensor.sinc_()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
a.sinc_()

# Paddle 写法
import numpy
paddle.assign(paddle.where(a==0, x=paddle.to_tensor([1], dtype=a.dtype), y=paddle.sin(numpy.pi*a)/(numpy.pi*a)), a)
```
