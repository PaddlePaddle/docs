## [ torch 参数更多 ]torch.diff
### [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff)

```python
torch.diff(input,
           n=1,
           dim=-1,
           prepend=None,
           append=None,
           *,
           out=None)
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

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| n           | n         | 表示需要计算前向差值的次数。 |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| prepend           | prepend         | 表示在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的前面。 |
| append           | append         | 表示在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的后面。 |
| out        | -  | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.diff(torch.tensor([3, 5]), out=y)

# Paddle 写法
paddle.assign(paddle.diff(paddle.to_tensor([3, 5])), y)
```
