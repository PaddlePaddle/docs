## [ 仅参数名不一致 ] torch.Tensor.nanquantile

### [torch.Tensor.nanquantile](https://pytorch.org/docs/stable/generated/torch.nanquantile.html#torch.nanquantile)

```python
torch.nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None)
```

### [paddle.Tensor.nanquantile](https://github.com/PaddlePaddle/Paddle/pull/41343)

```python
paddle.Tensor.nanquantile(x, q, axis=None, keepdim=False)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| input         | x            | 一个 多维 `Tensor`，数据类型为 `float16` 、 `float32` 、 `float64` 、 `int32` 或 `int64` ，仅参数名不一致。 |
| q             | q            | 一个 [0, 1] 范围内的分位数值的标量或一维张量，仅参数名不一致。 |
| dim           | axis         | 求乘积运算的维度，仅参数名不一致。                           |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留输入的维度，仅参数名不一致。         |
| interpolation | -            | 指定当所需分位数位于两个数据点之间时使用的插值方法，Paddle 无此功能，暂无转写方式。 |
| out           | -            | 一般无需设置，默认值为 None。                                |
