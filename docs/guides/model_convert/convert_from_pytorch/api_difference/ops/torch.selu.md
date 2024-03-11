## [仅参数名不一致]torch.selu

### [torch.selu](https://pytorch.org/docs/stable/generated/torch.nn.functional.selu.html#torch.nn.functional.selu)

```python
torch.selu(input)
```

### [paddle.nn.functional.selu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/selu_cn.html)

```python
paddle.nn.functional.selu(x, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
