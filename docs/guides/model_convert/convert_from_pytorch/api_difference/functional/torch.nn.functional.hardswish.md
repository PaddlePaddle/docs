## [torch 参数更多]torch.nn.functional.hardswish

### [torch.nn.functional.hardswish](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardswish.html#torch.nn.functional.hardswish)

```python
torch.nn.functional.hardswish(input, inplace=False)
```

### [paddle.nn.functional.hardswish](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/hardswish_cn.html)

```python
paddle.nn.functional.hardswish(x, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
