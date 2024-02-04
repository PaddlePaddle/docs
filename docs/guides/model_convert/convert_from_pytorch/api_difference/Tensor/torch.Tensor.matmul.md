## [ 仅参数名不一致 ]torch.Tensor.matmul

### [torch.Tensor.matmul](https://pytorch.org/docs/stable/generated/torch.Tensor.matmul.html)

```python
torch.Tensor.matmul(other)
```

### [paddle.Tensor.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#matmul-y-transpose-x-false-transpose-y-false-name-none)

```python
paddle.Tensor.matmul(y, transpose_x=False, transpose_y=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| other   | y            |  输入变量。 |
