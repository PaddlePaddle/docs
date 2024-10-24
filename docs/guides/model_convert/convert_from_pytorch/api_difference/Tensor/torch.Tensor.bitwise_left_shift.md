## [ paddle 参数更多 ]torch.Tensor.bitwise_left_shift

### [torch.Tensor.bitwise_left_shift](https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_left_shift.html#torch-tensor-bitwise-left-shift)

```python
torch.Tensor.bitwise_left_shift(other)
```

### [paddle.Tensor.bitwise_left_shift]()

```python
paddle.Tensor.bitwise_left_shift(y, is_arithmetic=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle  | 备注                                                                |
| ------- | ------------- | ------------------------------------------------------------------- |
| other   | y             | 表示输入的 Tensor ，仅参数名不一致。                                |
| -       | is_arithmetic | 用于表明是否执行算术位移， PyTorch 无此参数， Paddle 保持默认即可。 |

### 转写示例

```python
# PyTorch 写法
out = x.bitwise_left_shift(y)

# Paddle 写法
out = x.bitwise_left_shift(y)
```
