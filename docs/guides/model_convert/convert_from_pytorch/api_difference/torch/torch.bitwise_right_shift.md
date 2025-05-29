## [ paddle 参数更多 ]torch.bitwise_right_shift

### [torch.bitwise_right_shift](https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift.html)

```python
torch.bitwise_right_shift(input, other, *, out=None)
```

### [paddle.bitwise_right_shift](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bitwise_right_shift_cn.html#bitwise-right-shift)

```python
paddle.bitwise_right_shift(x, y, is_arithmetic=True, out=None, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle  | 备注                                                                |
| ------- | ------------- | ------------------------------------------------------------------- |
| input   | x             | 表示输入的 Tensor ，仅参数名不一致。                                |
| other   | y             | 表示输入的 Tensor ，仅参数名不一致。                                |
| out     | out           | 表示输出的 Tensor。                                                 |
| -       | is_arithmetic | 用于表明是否执行算术位移， PyTorch 无此参数， Paddle 保持默认即可。 |
