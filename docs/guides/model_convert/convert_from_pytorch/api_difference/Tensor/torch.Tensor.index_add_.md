## [ torch 参数更多 ]torch.Tensor.index_add_
### [torch.Tensor.index_add_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_)

```python
torch.Tensor.index_add_(dim, index, source, *, alpha=1)
```

### [paddle.index_add_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_add__cn.html)

```python
paddle.index_add_(x, index, axis, value, name=None)
```

其中 Pytorch 与 Paddle 参数有差异，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| - | <font color='red'> x </font> | 表示输入的 Tensor 。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| <font color='red'> index </font> | <font color='red'> index </font> | 包含索引下标的 1-D Tensor。  |
| <font color='red'> source </font> | <font color='red'> value </font> | 被加的 Tensor，仅参数名不一致。  |
| <font color='red'> alpha </font> | - | source 的 缩放倍数， Paddle 无此参数，需要进行转写。Paddle 应将 alpha 和 source 的乘积作为 value。 |


### 转写示例
#### alpha：source 的缩放倍数
```python
# Pytorch 写法
x.index_add_(dim=1, index=index, source=source, alpha=alpha)

# Paddle 写法
paddle.index_add_(x, index=index, axis=1, value=alpha*source)
```
