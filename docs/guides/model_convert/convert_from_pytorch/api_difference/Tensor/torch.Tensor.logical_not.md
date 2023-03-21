## [ 仅参数名不一致 ]torch.Tensor.logical_not

同 torch.logical_not

### [torch.Tensor.logical_not](https://pytorch.org/docs/stable/generated/torch.logical_not.html)

```python
torch.Tensor.logical_not(input, 
                         *, 
                         out=None)
```

### [paddle.Tensor.logical_not](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logical_not_cn.html#logical-not)

```python
paddle.Tensor.logical_not(x, 
                          out=None, 
                          name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ---------------------------------- |
| input   | x            | 输入的多维 Tensor ，仅参数名不同。 |
