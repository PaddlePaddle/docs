## [ 输入参数用法不一致 ]torch.diff
### [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff)

```python
torch.diff(input,
           n=1,
           dim=-1,
           prepend=None,
           append=None)
```

### [paddle.diff](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diff_cn.html#diff)

```python
paddle.diff(x,
            n=1,
            axis=-1,
            prepend=None,
            append=None,
            name=None)
```

两者功能一致，但参数 `n` 的支持范围不同，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| n           | n         | 表示需要计算前向差值的次数。torch 支持 n 为任意值，paddle 仅支持 n=1，暂无转写方式。 |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| prepend           | prepend         | 表示在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的前面。 |
| append           | append         | 表示在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的后面。 |
