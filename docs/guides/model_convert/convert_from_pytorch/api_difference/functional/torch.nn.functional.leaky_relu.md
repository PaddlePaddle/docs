## [torch 参数更多]torch.nn.functional.leaky_relu

### [torch.nn.functional.leaky_relu](https://pytorch.org/docs/1.13/generated/torch.nn.functional.leaky_relu.html#torch.nn.functional.leaky_relu)

```python
torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
```

### [paddle.nn.functional.leaky_relu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/leaky_relu_cn.html)

```python
paddle.nn.functional.leaky_relu(x, negative_slope=0.01, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle   | 备注                                                                                                            |
| -------------- | -------------- | --------------------------------------------------------------------------------------------------------------- |
| input          | x              | 输入的 Tensor，仅参数名不一致。                                                                                 |
| negative_slope | negative_slope | x<0 时的斜率。                                                                                                  |
| inplace        | -              | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
