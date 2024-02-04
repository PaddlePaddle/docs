## [ 仅参数名不一致 ]torch.Tensor.argmax

### [torch.Tensor.argmax](https://pytorch.org/docs/stable/generated/torch.Tensor.argmax.html)

```python
torch.Tensor.argmax(dim=None, keepdim=False)
```

### [paddle.Tensor.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#argmax-axis-none-keepdim-false-dtype-int64-name-none)

```python
paddle.Tensor.argmax(axis=None, keepdim=False, dtype=int64, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ------------------                 |
| dim     | axis         | 指定对输入 Tensor 进行运算的轴，axis 的有效范围是\[-R, R），R 是输入 x 的维度个数  |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
