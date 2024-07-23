## [ 输入参数用法不一致 ]torch.slice_scatter

### [torch.slice_scatter](https://pytorch.org/docs/stable/generated/torch.slice_scatter.html#torch.slice_scatter)

```python
torch.slice_scatter(input, src, dim=0, start=None, end=None, step=1)
```

### [paddle.slice_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/slice_scatter.html)

```python
paddle.slice_scatter(x, value, axes, starts, ends, strides, name=None)
```

两者功能一致，参数不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的目标矩阵, 仅参数名不一致。 |
| src           | value        | 嵌入的值，仅参数名不一致。 |
| dim           | axes         | 嵌入的维度，PyTorch 为 int 类型，Paddle 为 list of int。 |
| start         | starts       | 嵌入起始索引，PyTorch 为 int 类型，Paddle 为 list of int。 |
| end           | ends         | 嵌入截至索引，PyTorch 为 int 类型，Paddle 为 list of int。 |
| step          | strides      | 嵌入步长，PyTorch 为 int 类型，Paddle 为 list of int。 |

### 转写示例

```python
# PyTorch 写法
torch.slice_scatter(input, src, dim=0, start=1, end=5, step=2)

# Paddle 写法
paddle.slice_scatter(x, value, axes=[0], starts=[1], ends=[5], strides=[2])
```
