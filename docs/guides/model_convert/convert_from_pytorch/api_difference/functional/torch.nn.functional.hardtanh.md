## [torch 参数更多]torch.nn.functional.hardtanh

### [torch.nn.functional.hardtanh](https://pytorch.org/docs/1.13/generated/torch.nn.functional.hardtanh.html#torch.nn.functional.hardtanh)

```python
torch.nn.functional.hardtanh(input, min_val=- 1.0, max_val=1.0, inplace=False)
```

### [paddle.nn.functional.hardtanh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/hardtanh_cn.html)

```python
paddle.nn.functional.hardtanh(x, min=-1.0, max=1.0, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| min_val | min          | hardtanh 激活计算公式中的 min 值，仅参数名不一致。                                                              |
| max_val | max          | hardtanh 激活计算公式中的 max 值，仅参数名不一致。                                                              |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
