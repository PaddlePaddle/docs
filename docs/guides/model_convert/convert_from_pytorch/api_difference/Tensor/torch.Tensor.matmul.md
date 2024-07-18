## [ paddle 参数更多 ]torch.Tensor.matmul

### [torch.Tensor.matmul](https://pytorch.org/docs/stable/generated/torch.Tensor.matmul.html)

```python
torch.Tensor.matmul(other)
```

### [paddle.Tensor.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#matmul-y-transpose-x-false-transpose-y-false-name-none)

```python
paddle.Tensor.matmul(y, transpose_x=False, transpose_y=False, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | ---- |
| other   | y            |  输入变量，仅参数名不一致。 |
| -       | transpose_x  |  相乘前是否转置 x，PyTorch 无此参数，Paddle 保持默认即可。 |
| -       | transpose_y  |  相乘前是否转置 y，PyTorch 无此参数，Paddle 保持默认即可。 |
