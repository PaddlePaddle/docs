## [ 仅参数名不一致 ]torch.any

### [torch.any](https://pytorch.org/docs/stable/generated/torch.any.html?highlight=any#torch.any)

```python
torch.any(input)
```

### [paddle.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/any_cn.html#any)

```python
paddle.any(x,
           axis=None,
           keepdim=False,
           name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>        | <font color='red'> x </font>           | 输入的多维 Tensor ，仅参数名不同。                   |
| -    | <font color='red'> axis </font>     | 计算逻辑与运算的维度，Pytorch 无此参数，保持默认即可。        |
| -    | <font color='red'> keepdim </font>| 是否在输出 Tensor 中保留减小的维度，Pytorch 无此参数，保持默认即可。  |
