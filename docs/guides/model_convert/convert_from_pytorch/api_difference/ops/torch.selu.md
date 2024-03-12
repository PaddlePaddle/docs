## [仅 paddle 参数更多]torch.selu

### [torch.selu](https://pytorch.org/docs/stable/generated/torch.nn.functional.selu.html#torch.nn.functional.selu)

```python
torch.selu(input)
```

### [paddle.nn.functional.selu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/selu_cn.html)

```python
paddle.nn.functional.selu(x, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| -       | scale        | selu 激活计算公式中的 scale 值，PyTorch 无此参数，Paddle 保持默认即可。 |
| -       | alpha        | selu 激活计算公式中的 alpha 值，PyTorch 无此参数，Paddle 保持默认即可。 |
