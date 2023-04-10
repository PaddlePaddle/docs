## [ 仅参数名不一致 ]torch.prod
### [torch.prod](https://pytorch.org/docs/stable/generated/torch.prod.html?highlight=prod#torch.prod)


```python
torch.prod(input,
           dim=None,
           keepdim=False,
           dtype=None)
```

### [paddle.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/prod_cn.html#prod)

```python
paddle.prod(x,
            axis=None,
            keepdim=False,
            dtype=None,
            name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行乘积运算的轴，仅参数名不一致。  |
| keepdim           | keepdim         | 表示是否在输出 Tensor 中保留减小的维度。                  |
| dtype           | dtype         | 表示数据类型。                  |
