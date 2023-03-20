## [ 仅参数名不一致 ]torch.Tensor.logical_or

同torch.logical_or

### [torch.Tensor.logical_or](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_or.html)

```python
torch.Tensor.logical_or(input, 
                        other, 
                        *, 
                        out=None)
```

### [paddle.Tensor.logical_or](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logical_or_cn.html#logical-or)

```python
paddle.Tensor.logical_or(x, 
                         y, 
                         out=None, 
                         name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                          |
| ------- | ------------ | --------------------------------------------- |
| input   | x            | 第一个参与逻辑或运算的Tensor ，仅参数名不同。 |
| other   | y            | 第二个参与逻辑或运算的Tensor ，仅参数名不同。 |