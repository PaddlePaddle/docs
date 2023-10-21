## [ 组合替代实现 ]torch.float_power

### [torch.float_power](https://pytorch.org/docs/stable/generated/torch.float_power.html#torch-float-power)
```python
torch.float_power(input, exponent)
```

以双倍精度将输入以元素为单位提升到指数的幂级数。如果两个输入都不是复数，则返回 torch.float64 张量；
如果一个或多个输入是复数，则返回 torch.complex128 张量。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API，不过paddle.cast不支持复数类型的转换。

###  转写示例

```python
# Pytorch 写法
import paddle

y = torch.float_power(input, exponent)

# Paddle 写法
y = paddle.cast(paddle.pow(input, exponent), 'float64')
```
