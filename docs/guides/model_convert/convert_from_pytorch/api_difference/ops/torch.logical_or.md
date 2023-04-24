## [ 仅参数名不一致 ]torch.logical_or
### [torch.logical_or](https://pytorch.org/docs/1.13/generated/torch.logical_or.html?highlight=logical_or#torch.logical_or)

```python
torch.logical_or(input,
                 other,
                 *,
                 out=None)
```

### [paddle.logical_or](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logical_or_cn.html#logical-or)

```python
paddle.logical_or(x,
                  y,
                  out=None,
                  name=None)
```



两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> other </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| out | out | 表示输出的 Tensor 。 |
