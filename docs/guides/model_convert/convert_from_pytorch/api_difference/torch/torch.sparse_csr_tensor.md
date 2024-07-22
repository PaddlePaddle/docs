## [torch 参数更多]torch.sparse_csr_tensor

### [torch.sparse_csr_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html#torch.sparse_csr_tensor)

```python
torch.sparse_csr_tensor(crow_indices,
                        col_indices,
                        values,
                        size=None,
                        *,
                        dtype=None,
                        layout=torch.strided,
                        device=None,
                        pin_memory=False,
                        requires_grad=False,
                        check_invariants=None)
```

### [paddle.sparse.sparse_csr_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/sparse_csr_tensor_cn.html#sparse-csr-tensor)

```python
paddle.sparse.sparse_csr_tensor(crows, cols, values, shape, dtype=None, place=None, stop_gradient=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch          | PaddlePaddle  | 备注                                                           |
| ---------------- | ------------- | -------------------------------------------------------------- |
| crow_indices     | crows         | 每行第一个非零元素在 values 的起始位置，仅参数名不一致。       |
| col_indices      | cols          | 一维数组，存储每个非零元素的列信息，仅参数名不一致。           |
| values           | values        | 一维数组，存储非零元素。                                       |
| size             | shape         | 稀疏 Tensor 的形状，仅参数名不一致。                           |
| dtype            | dtype         | 创建 tensor 的数据类型。                                       |
| layout           |-              |表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| device           | place         | 创建 tensor 的设备位置，仅参数名不一致。                       |
| pin_memory       | -             | 表示是否使用锁页内存， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| requires_grad    | stop_gradient | 是否阻断 Autograd 的梯度传导，两者参数功能相反，需要转写。 |
| check_invariants | -             | 是否检查稀疏 Tensor 变量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### requires_grad 参数：是否阻断 Autograd 的梯度传导

```python
# PyTorch 写法
crows = [0, 2, 3, 5]
cols = [1, 3, 2, 0, 1]
values = [1, 2, 3, 4, 5]
dense_shape = [3, 4]
csr = torch.sparse_csr_tensor(crows, cols, values, dense_shape, requires_grad=False)

# Paddle 写法
crows = [0, 2, 3, 5]
cols = [1, 3, 2, 0, 1]
values = [1, 2, 3, 4, 5]
dense_shape = [3, 4]
csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape, stop_gradient= True)
```
