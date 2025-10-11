## [torch 参数更多]torch.nn.functional.celu

### [torch.nn.functional.celu](https://pytorch.org/docs/stable/generated/torch.nn.functional.celu.html#torch.nn.functional.celu)

```python
torch.nn.functional.celu(input, alpha=1.0, inplace=False)
```

### [paddle.nn.functional.celu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/celu_cn.html)

```python
paddle.nn.functional.celu(x, alpha=1.0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| alpha   | alpha        | alpha 参数。                                                                                                    |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
