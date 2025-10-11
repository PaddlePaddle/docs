## [仅参数名不一致]torch.special.log_softmax

### [torch.special.log_softmax](https://pytorch.org/docs/stable/special.html#torch.special.log_softmax)

```python
torch.special.log_softmax(input, dim, *, dtype=None)
```

### [paddle.nn.functional.log_softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/log_softmax_cn.html)

```python
paddle.nn.functional.log_softmax(x, axis=- 1, dtype=None, name=None)
```

其中功能一致, 仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                        |
| ------- | ------------ | ------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。             |
| dim     | axis         | 指定对输入 x 进行运算的轴，仅参数名不一致。 |
| dtype   | dtype        | 输出 Tensor 的数据类型。                    |
