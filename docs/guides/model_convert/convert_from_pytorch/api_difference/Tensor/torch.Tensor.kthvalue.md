## [ 仅参数名不一致 ]torch.Tensor.kthvalue

### [torch.Tensor.kthvalue](https://pytorch.org/docs/stable/generated/torch.Tensor.kthvalue.html)

```python
torch.Tensor.kthvalue(k, dim=None, keepdim=False)
```

### [paddle.Tensor.kthvalue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#kthvalue-k-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.kthvalue(k, axis=None, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| k       | k            | 需要沿轴查找的第 k 小，所对应的 k 值。 |
| dim     | axis         | 指定对输入 Tensor 进行运算的轴，axis 的有效范围是[-R, R），R 是输入 x 的 Rank， axis 为负时与 axis + R 等价。默认值为-1。 |
| keepdim | keepdim      | 是否保留指定的轴。如果是 True，维度会与输入 x 一致，对应所指定的轴的 size 为 1。否则，由于对应轴被展开，输出的维度会比输入小 1。默认值为 False。 |
