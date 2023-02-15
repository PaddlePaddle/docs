## torch.gather
### [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather)

```python
torch.gather(input,
                dim,
                index,
                *,
                sparse_grad=False,
                out=None)
```

### [paddle.take_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/take_along_axis_cn.html#take-along-axis)

```python
paddle.take_along_axis(arr,
                        indices,
                        axis)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。                                    |
| dim           | axis         | 用于指定 index 获取输入的维度，PyTorch 中类型仅能为 int，PaddlePaddle 中类型可以为 int32/int64/Tensor。 |
| index         | indices      | 聚合元素的索引矩阵。                                   |
| sparse_grad   | -            | 表示是否对梯度稀疏化，PaddlePaddle 无此参数。            |
| out           | -            | 表示目标 Tensor，PaddlePaddle 无此参数。               |

### 功能差异
#### 使用方式
***PyTorch***：索引(index)的维度数和输入(input)的维度数一致，索引(index)的形状大小要小于等于输入(input)的形状大小。
***PaddlePaddle***：索引(indices)的秩有且只能等于 1。

#### 计算方式
***PyTorch***：沿指定的轴(dim)收集值。以 2-D Tensor 输入为例，其输出结果如下:
```
if dim == 0:
    out[i][j] = input[index[i][j]][j]
if dim == 1:
    out[i][j] = input[i][index[i][j]]
```
***PaddlePaddle***：根据索引(index)获取输入(x)的指定维度(axis)的条目，并将它们拼接在一起。以 2-D Tensor 输入为例，其输出结果如下:
```
if axis == 0:
    tensor_list = list()
    for i in index:
        tensor_list.append(index[i, :])
    将 tensor_list 中的 tensor 沿 axis 轴拼接
if axis == 1:
    tensor_list = list()
    for i in index:
        tensor_list.append(index[:, i])
    将 tensor_list 中的 tensor 沿 axis 轴拼接
```

### 代码示例

``` python
# PyTorch 示例：
t = torch.tensor([[1, 2], [3, 4]])
torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
# 输出
# tensor([[ 1,  1],
#         [ 4,  3]])
```

``` python
# PaddlePaddle 示例：
t = paddle.to_tensor([[1, 2], [3, 4]])
paddle.gather(t, paddle.to_tensor([1, 0]), 1)
# 输出
# Tensor(shape=[2, 2], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[2, 1],
#         [4, 3]])
```

### 组合实现

```python
def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

t = paddle.to_tensor([[1, 2], [3, 4]])
paddle_gather(t, 1, paddle.to_tensor([[0, 0], [1, 0]]))
# 输出
# Tensor(shape=[2, 2], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[1, 1],
#         [4, 3]])
```
