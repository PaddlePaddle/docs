## [ torch 参数更多 ] torch.Tensor.nanquantile

### [torch.Tensor.nanquantile](https://pytorch.org/docs/stable/generated/torch.nanquantile.html#torch.nanquantile)

```python
torch.Tensor.nanquantile(q, dim=None, keepdim=False, *, interpolation='linear')
```

### [paddle.Tensor.nanquantile](https://github.com/PaddlePaddle/Paddle/pull/41343)

```python
paddle.Tensor.nanquantile(q, axis=None, keepdim=False)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| q             | q            | 一个 [0, 1] 范围内的分位数值的标量或一维张量。                      |
| dim           | axis         | 求乘积运算的维度，仅参数名不一致。                                                  |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留输入的维度。                                |
| interpolation | -            | 指定当所需分位数位于两个数据点之间时使用的插值方法，Paddle 无此功能，暂无转写方式。 |


### 转写示例

```python
# Pytorch 写法
x = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
x.nanquantile(0.6, dim=0)

# Paddle 写法
x = paddle.to_tensor([0., 1., 2., 3.], dtype='float64')
x.nanquantile(0.6, axis=0)
```
