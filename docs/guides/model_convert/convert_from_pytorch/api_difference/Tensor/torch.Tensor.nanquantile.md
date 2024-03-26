## [ torch 参数更多 ] torch.Tensor.nanquantile

### [torch.Tensor.nanquantile](https://pytorch.org/docs/stable/generated/torch.nanquantile.html#torch.nanquantile)

```python
torch.Tensor.nanquantile(q, dim=None, keepdim=False, *, interpolation='linear')
```

### [paddle.Tensor.nanquantile](https://github.com/PaddlePaddle/Paddle/pull/41343)

```python
paddle.Tensor.nanquantile(q, axis=None, keepdim=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                                                                |
| ------------- | ------------ |-----------------------------------------------------------------------------------------------------------------------------------|
| q             | q            | 待计算的分位数。|
| dim           | axis         | 求乘积运算的维度，仅参数名不一致。                                                                                                                 |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留输入的维度。                                                                                                            |
<<<<<<< HEAD
| interpolation | interpolation | 指定当所需分位数位于两个数据点之间时使用的插值方法。|
=======
| interpolation | -            | 指定当所需分位数位于两个数据点之间时使用的插值方法，Paddle 无此参数，暂无转写方式。                                                                                     |

### 转写示例
```python
# 当 q 为向量时
# PyTorch 写法
x.nanquantile(q=torch.tensor([0.5, 0.1], dtype=torch.float64))

# Paddle
x.nanquantile(q=[0.5, 0.1])
```
>>>>>>> up/develop
