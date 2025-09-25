## [ 仅参数名不一致 ]torch.Tensor.sum

### [torch.Tensor.sum](https://pytorch.org/docs/stable/generated/torch.Tensor.sum.html)

```python
torch.Tensor.sum(dim=None, keepdim=False, dtype=None)
```

### [paddle.Tensor.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#sum-axis-none-dtype-none-keepdim-false-name-none)
```python
paddle.Tensor.sum(axis=None, dtype=None, keepdim=False, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------|
| dim           | axis         |  求和运算的维度，仅参数名不一致。                                       |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留减小的维度 。                      |
| dtype         | dtype        | 是输出变量的数据类型 。                                   |
