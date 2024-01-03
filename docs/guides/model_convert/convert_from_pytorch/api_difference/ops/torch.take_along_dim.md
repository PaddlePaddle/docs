## [ torch 参数更多 ]torch.take_along_dim

### [torch.take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html?highlight=torch+take_along_dim#torch.take_along_dim)

```python
torch.take_along_dim(input,
                     indices,
                     dim,
                     out=None)
```

### [paddle.take_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/take_along_axis_cn.html)

```python
paddle.take_along_axis(arr,
                       indices,
                       axis,
                       broadcast=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | arr     | 表示输入的 Tensor ，仅参数名不一致。                                     |
| indices    | indices   | 表示索引矩阵 ，仅参数名不一致。                              |
| dim        | axis      | 表示沿着哪个维度获取对应的值，仅参数名不一致。                 |
| out    | - | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.take_along_dim(t, idx, 1，out=y)

# Paddle 写法
paddle.assign(paddle.take_along_axis(t, idx, 1), y)
```
