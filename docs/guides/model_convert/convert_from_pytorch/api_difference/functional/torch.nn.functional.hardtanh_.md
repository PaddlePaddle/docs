## [仅参数名不一致]torch.nn.functional.hardtanh_

### [torch.nn.functional.hardtanh_](https://pytorch.org/docs/stable/jit_builtin_functions.html#supported-tensor-methods)

```python
torch.nn.functional.hardtanh_(input, min_val=-1, max_val=1)
```

### [paddle.nn.functional.hardtanh_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hardtanh_cn.html)

```python
paddle.nn.functional.hardtanh_(x, min=-1.0, max=1.0, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| min_val | min          | hardtanh 激活计算公式中的 min 值，仅参数名不一致。                                                              |
| max_val | max          | hardtanh 激活计算公式中的 max 值，仅参数名不一致。                                                              |
