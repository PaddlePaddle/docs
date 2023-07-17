## [torch 参数更多]torch.nn.Hardtanh

### [torch.nn.Hardtanh](https://pytorch.org/docs/1.13/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh)

```python
torch.nn.Hardtanh(min_val=- 1.0, max_val=1.0, inplace=False)
```

### [paddle.nn.Hardtanh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Hardtanh_cn.html)

```python
paddle.nn.Hardtanh(min=- 1.0, max=1.0, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                        |
| ------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| min_val | min          | Hardtanh 激活计算公式中的 min 值，仅参数名不一致。                                                          |
| max_val | max          | Hardtanh 激活计算公式中的 max 值，仅参数名不一致。                                                          |
| inplace | -            | 在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
