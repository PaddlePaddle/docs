## [ 仅参数名不一致 ]torch.celu

### [torch.celu](https://pytorch.org/docs/stable/generated/torch.nn.functional.celu.html#torch.nn.functional.celu)

```python
torch.celu(input, alpha=1.0)
```

### [paddle.nn.functional.celu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/celu_cn.html#celu)

```python
paddle.nn.functional.celu(x, alpha=1.0, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入的 Tensor。  |
| alpha   | alpha        | CELU 的 alpha 值，默认值为 1.0。 |
