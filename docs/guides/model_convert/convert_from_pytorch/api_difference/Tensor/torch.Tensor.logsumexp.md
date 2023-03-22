## [ 仅参数名不⼀致 ] torch.Tensor.logsumexp


### [torch.Tensor.logsumexp](https://pytorch.org/docs/stable/generated/torch.Tensor.logsumexp.html)

```python
torch.Tensor.logsumexp(dim, 
                       keepdim=False) 
```

### [paddle.Tensor.logsumexp](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/Tensor_cn.html#logsumexp-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.logsumexp(axis=None, 
                        keepdim=False, 
                        name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| dim   | axis           | 表示进行运算的轴，仅参数名不一致。                       |
| keepdim     | keepdim         | 是否在输出 Tensor 中保留减小的维度                       |
