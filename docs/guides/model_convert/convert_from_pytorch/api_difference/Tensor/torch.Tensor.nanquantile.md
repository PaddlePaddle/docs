## [ 仅参数名不一致 ] torch.Tensor.nanquantile

### [torch.Tensor.nanquantile](https://pytorch.org/docs/stable/generated/torch.nanquantile.html#torch.nanquantile)

```python
torch.Tensor.nanquantile(q, dim=None, keepdim=False, *, interpolation='linear', out=None)
```

### [paddle.Tensor.nanquantile](https://github.com/PaddlePaddle/Paddle/pull/41343)

```python
paddle.Tensor.nanquantile(q, axis=None, keepdim=False)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| q             | q            | 一个 [0, 1] 范围内的分位数值的标量或一维张量，仅参数名不一致。 |
| dim           | axis         | 求乘积运算的维度，仅参数名不一致。                           |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留输入的维度，仅参数名不一致。         |
| interpolation | -            | 指定当所需分位数位于两个数据点之间时使用的插值方法，Paddle 无此功能，暂无转写方式。 |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。                              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.Tensor.nanquantile(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]), out = y) # 同 y = torch.Tensor.nanquantile(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
# Paddle 写法
y = paddle.Tensor.nanquantile(paddle.to_tensor([[1, 2], [3, 4]]), paddle.to_tensor([[1, 1], [4, 4]]))
```
