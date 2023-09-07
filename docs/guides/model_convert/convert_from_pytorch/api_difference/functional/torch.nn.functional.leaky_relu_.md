## [仅参数名不一致]torch.nn.functional.leaky_relu_

### [torch.nn.functional.leaky_relu_](https://pytorch.org/docs/stable/jit_builtin_functions.html#supported-tensor-methods)

```python
torch.nn.functional.leaky_relu_(input, negative_slope=0.01)
```

### [paddle.nn.functional.leaky_relu_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/leaky_relu_cn.html)

```python
paddle.nn.functional.leaky_relu_(x, negative_slope=0.01)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle   | 备注                                                                                                            |
| -------------- | -------------- | --------------------------------------------------------------------------------------------------------------- |
| input          | x              | 输入的 Tensor，仅参数名不一致。                                                                                 |
| negative_slope | negative_slope | x<0 时的斜率。                                                                                                  |
