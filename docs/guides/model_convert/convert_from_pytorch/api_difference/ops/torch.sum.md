## [ 仅参数名不一致 ]torch.sum
### [torch.sum](https://pytorch.org/docs/stable/generated/torch.sum.html?highlight=sum#torch.sum)

```python
torch.sum(input,
          dim=None,
          keepdim=False,
          dtype=None)
```

### [paddle.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sum_cn.html#sum)

```python
paddle.sum(x,
           axis=None,
           dtype=None,
           keepdim=False,
           name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| keepdim           | keepdim         | 是否在输出 Tensor 中保留减小的维度。 |
| dtype           | dtype         | 表示数据类型。 |
