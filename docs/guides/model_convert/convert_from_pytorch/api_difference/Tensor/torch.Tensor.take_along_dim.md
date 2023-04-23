## [ torch 参数更多 ]torch.Tensor.take_along_dim
### [torch.Tensor.take_along_dim](https://pytorch.org/docs/1.13/generated/torch.Tensor.take_along_dim.html?highlight=torch+tensor+take_along_dim#torch.Tensor.take_along_dim)

```python
torch.Tensor.take_along_dim(input,
                    indices,
                    dim,
                    *,
                    out=None)
```

### [paddle.Tensor.take_along_axis]( )

```python
paddle.Tensor.take_along_axis(indices,
                    axis)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | -            | 输入 Tensor 作为源矩阵， Paddle 无此参数，需要进行转写。  |
| indices         | indices         | 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。 |
|dim         | axis         |   指定沿着哪个维度获取对应的值，数据类型为：int，仅参数名不一致。 |
| out   |   -   |     表示输出的 Tensor ， Paddle 无此参数，需要进行转写。                              |

### 转写示例
#### input：输入
```python
# Pytorch 写法
x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
index = paddle.to_tensor([[0]])
torch.Tensor.take_along_dim(x, index, 0, out = y)

# Paddle 写法
x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
index = paddle.to_tensor([[0]])
y = x.take_along_axis(index, 0)
```

#### out：指定输出
```python
# Pytorch 写法
torch.Tensor.take_along_dim(x, index, 0, out = y)

# Paddle 写法
y = x.take_along_axis(index, 0)
```
