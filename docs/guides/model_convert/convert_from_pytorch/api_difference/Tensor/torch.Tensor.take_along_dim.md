## [ torch 参数更多 ]torch.take_along_dim
### [torch.take_along_dim](https://pytorch.org/docs/1.13/generated/torch.take_along_dim.html#torch.take_along_dim)

```python
torch.take_along_dim(input,
                    indices,
                    dim,
                    *,
                    out=None)
```

### [paddle.take_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/take_along_axis_cn.html#take-along-axis)

```python
paddle.take_along_axis(arr,
                    indices,
                    axis)
```

两者功能一致但参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | arr            | 输入的 Tensor 作为源矩阵，数据类型为：float32、float64，仅参数名不一致。  |
| indices         | indices         | 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。 |
|dim         | axis         |   指定沿着哪个维度获取对应的值，数据类型为：int，仅参数名不一致。 |
| out   |   -   |     表示输出的 Tensor ， Paddle 无此参数，需要进行转写。                              |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
index = torch.tensor([[0]])
axis = 0
torch.take_along_dim(x, index, axis, out = y)

# Paddle 写法
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
index = paddle.to_tensor([[0]])
axis = 0
paddle.assign(paddle.take_along_axis(x, index, axis), y)
```
