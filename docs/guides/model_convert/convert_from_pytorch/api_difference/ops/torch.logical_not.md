## [ 仅参数名不一致 ]torch.logical_not
### [torch.logical_not](https://pytorch.org/docs/stable/generated/torch.logical_not.html?highlight=logical_not#torch.logical_not)

```python
torch.logical_not(input,
                  *,
                  out=None)
```

### [paddle.logical_not](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logical_not_cn.html#logical-not)

```python
paddle.logical_not(x,
                   out=None,
                   name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| out | out | 表示输出的 Tensor 。  |
