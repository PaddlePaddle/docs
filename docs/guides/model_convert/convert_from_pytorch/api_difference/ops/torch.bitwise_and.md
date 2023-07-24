## [ 仅参数名不一致 ]torch.bitwise_and

### [torch.bitwise_and](https://pytorch.org/docs/stable/generated/torch.bitwise_and.html#torch.bitwise_and)

```python
torch.bitwise_and(input,
                  other,
                  *,
                  out=None)
```

### [paddle.bitwise_and](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bitwise_and_cn.html#bitwise-and)

```python
paddle.bitwise_and(x,
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
