## [ 仅参数名不一致 ]torch.bitwise_xor

### [torch.bitwise_xor](https://pytorch.org/docs/stable/generated/torch.bitwise_xor.html)

```python
torch.bitwise_xor(input,
                  other,
                  *,
                  out=None)
```

### [paddle.bitwise_xor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bitwise_xor_cn.html)

```python
paddle.bitwise_xor(x,
                   y,
                   out=None,
                   name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input  |   x   | 表示输入的 Tensor ，仅参数名不一致。   |
| other  |   y   | 表示输入的 Tensor ，仅参数名不一致。 |
| out | out | 表示输出的 Tensor。|
