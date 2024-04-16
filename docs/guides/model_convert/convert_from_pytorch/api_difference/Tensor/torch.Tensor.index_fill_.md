## [ 仅参数名不一致 ]torch.Tensor.index_fill_
### [torch.Tensor.index_fill_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_fill_.html?highlight=index_fill_#torch.Tensor.index_fill_)

```python
torch.Tensor.index_fill_(dim, index, value)
```

### [paddle.Tensor.index_fill_]()

```python
paddle.Tensor.index_fill_(index, axis, value, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| <font color='red'> index </font> | <font color='red'> index </font> | 包含索引下标的 1-D Tensor。  |
| <font color='red'> value </font> | <font color='red'> value </font> | 填充的值。  |

### 转写示例
#### alpha：source 的缩放倍数
```python
# PyTorch 写法
x.index_fill_(dim=1, index=index, value=1)

# Paddle 写法
x.index_fill_(index=index, axis=1, value=1)
