## [torch 参数更多]torch.nn.functional.relu

### [torch.nn.functional.relu](https://pytorch.org/docs/1.13/generated/torch.nn.functional.relu.html#torch.nn.functional.relu)

```python
torch.nn.functional.relu(input, inplace=False)
```

### [paddle.nn.functional.relu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/relu_cn.html)

```python
paddle.nn.functional.relu(x, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
