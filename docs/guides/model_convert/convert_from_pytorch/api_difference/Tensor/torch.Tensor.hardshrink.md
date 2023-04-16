## [ torch 参数更多 ]torch.Tensor.hardshrink

### [torch.Tensor.hardshrink](https://pytorch.org/docs/1.13/generated/torch.Tensor.hardshrink.html?highlight=torch+tensor+hardshrink#torch.Tensor.hardshrink)

```python
torch.Tensor.hardshrink(input, lambd=0.5)
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
| lambd         | threshold            | Hardshrink 激活计算公式中的 threshold 值。默认值为 0.5。                   |
