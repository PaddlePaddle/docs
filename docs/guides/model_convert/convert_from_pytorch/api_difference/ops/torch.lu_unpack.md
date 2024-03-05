## [torch 参数更多 ]torch.lu_unpack
### [torch.lu_unpack](https://pytorch.org/docs/stable/generated/torch.lu_unpack.html?highlight=lu_unpack#torch.lu_unpack)

```python
torch.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None)
```

### [paddle.linalg.lu_unpack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_unpack_cn.html)

```python
paddle.linalg.lu_unpack(x, y, unpack_ludata=True, unpack_pivots=True, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> LU_data </font>        | <font color='red'> x </font>              | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> LU_pivots </font>      | <font color='red'> y </font>              | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> unpack_data </font>    | <font color='red'> unpack_ludata </font>  | 输入的 bool ，仅参数名不一致。                                     |
| <font color='red'> unpack_pivots </font>  | <font color='red'> unpack_pivots </font>  | 输入的 bool ，参数完全一致。             |
| <font color='red'> out </font>            | -                                         | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |

### 转写示例

#### out：指定输出
```python
# PyTorch 写法
torch.lu_unpack(*torch.lu(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])), out=(P, L, U))

# Paddle 写法
P,L,U = paddle.linalg.lu_unpack(*paddle.linalg.lu(paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
```
