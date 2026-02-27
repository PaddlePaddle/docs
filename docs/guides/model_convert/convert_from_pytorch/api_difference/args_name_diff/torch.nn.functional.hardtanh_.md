## [ 仅参数名不一致 ]torch.nn.functional.hardtanh_
### [torch.nn.functional.hardtanh\_](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.hardtanh_.html#torch.nn.functional.hardtanh_)
```python
torch.nn.functional.hardtanh_(input, min_val=-1, max_val=1)
```

### [paddle.nn.functional.hardtanh\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hardtanh__cn.html#paddle.nn.functional.hardtanh_)
```python
paddle.nn.functional.hardtanh_(x, min=-1.0, max=1.0)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| min_val | min          | hardtanh 激活计算公式中的 min 值，仅参数名不一致。                                                              |
| max_val | max          | hardtanh 激活计算公式中的 max 值，仅参数名不一致。                                                              |
