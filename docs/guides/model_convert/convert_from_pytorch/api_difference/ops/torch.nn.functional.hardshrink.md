## [ 仅参数名不一致 ]torch.nn.functional.hardshrink

### [torch.nn.functional.hardshrink](https://pytorch.org/docs/1.13/generated/torch.nn.functional.hardshrink.html#torch.nn.functional.hardshrink)

```python
torch.nn.functional.hardshrink(input, lambd=0.5)
```

### [paddle.nn.functional.hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/hardshrink_cn.html#hardshrink)

```python
paddle.nn.functional.hardshrink(x, threshold=0.5, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
