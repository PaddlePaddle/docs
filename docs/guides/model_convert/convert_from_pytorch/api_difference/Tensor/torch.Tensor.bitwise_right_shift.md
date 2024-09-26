## [ paddle参数更多 ]torch.Tensor.bitwise_right_shift

### [torch.Tensor.bitwise_right_shift](https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_right_shift.html#torch-tensor-bitwise-right-shift)

```python
torch.Tensor.bitwise_right_shift(other)
```

### [paddle.bitwise_right_shift](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bitwise_right_shift_cn.html#bitwise-right-shift)

```python
paddle.bitwise_right_shift(x, y, is_arithmetic=True, out=None, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle  | 备注                                                                |
| ------- | ------------- | ------------------------------------------------------------------- |
| other   | y             | 表示输入的 Tensor ，仅参数名不一致。                                |
| -       | out           | 表示输出的 Tensor，PyTorch 无此参数，保持默认即可。                 |
| -       | is_arithmetic | 用于表明是否执行算术位移， PyTorch 无此参数， Paddle 保持默认即可。 |

### 转写示例

```python
# PyTorch 写法
out = x.bitwise_right_shift(y)

# Paddle 写法
out = paddle.bitwise_right_shift(x, y)
```
