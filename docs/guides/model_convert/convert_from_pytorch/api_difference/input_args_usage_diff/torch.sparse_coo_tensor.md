## [ 输入参数用法不一致 ]torch.sparse_coo_tensor
### [torch.sparse\_coo\_tensor](https://docs.pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor)
```python
torch.sparse_coo_tensor(indices,values,size=None, * , dtype=None, device=None, requires_grad=False, check_invariants=None)
```

### [paddle.sparse.sparse\_coo\_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/sparse_coo_tensor_cn.html#paddle.sparse.sparse_coo_tensor)
```python
paddle.sparse.sparse_coo_tensor(indices, values, shape=None, dtype=None, place=None, stop_gradient=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

|    PyTorch    | PaddlePaddle  |                    备注                     |
|  -----------  |  -----------  |  ----------------------------------------- |
|    indices    |    indices    |         表示初始化 tensor 的数据。          |
|    values     |    values     |         表示初始化 tensor 的数据。          |
|     size      |     shape     |      表示张量的大小，仅参数名不一致。       |
|     dtype     |     dtype     |        表示创建 tensor 的数据类型。         |
|    device     |     place     |  表示创建 tensor 的设备位置，仅参数名不一致。 |
| requires_grad | stop_gradient |     两者参数功能相反，需要转写。      |
| check_invariants | -             | 是否检查稀疏 Tensor 变量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例
#### requires_grad 参数
```python
coo = torch.sparse_coo_tensor(indices, values, dense_shape，requires_grad=False)

# Paddle 写法
coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape，stop_gradient=True)
```
