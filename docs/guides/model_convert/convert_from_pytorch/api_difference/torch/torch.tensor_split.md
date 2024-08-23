## [ 仅参数名不一致 ]torch.tensor_split
api 存在重载情况，分别如下：

-------------------------------------------------------------------------------------------------

### [torch.tensor_split](https://pytorch.org/docs/stable/generated/torch.tensor_split.html)

```python
torch.tensor_split(input, indices, dim=0)
```

### [paddle.tensor_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tensor_split_cn.html)

```python
paddle.tensor_split(x, num_or_indices, axis=0, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
| indices           | num_or_indices         | 表示分割的索引，仅参数名不一致。                          |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。                          |

-------------------------------------------------------------------------------------------------

### [torch.tensor_split](https://pytorch.org/docs/stable/generated/torch.tensor_split.html)

```python
torch.tensor_split(input, tensor_indices_or_sections, dim=0)
```

### [paddle.tensor_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tensor_split_cn.html)

```python
paddle.tensor_split(x, num_or_indices, axis=0, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
| tensor_indices_or_sections           | num_or_indices         | 表示分割的数量或索引，仅参数名不一致。                          |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。                          |

-------------------------------------------------------------------------------------------------

### [torch.tensor_split](https://pytorch.org/docs/stable/generated/torch.tensor_split.html)

```python
torch.tensor_split(input, sections, dim=0)
```

### [paddle.tensor_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tensor_split_cn.html)

```python
paddle.tensor_split(x, num_or_indices, axis=0, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
| sections           | num_or_indices         | 表示分割的数量，仅参数名不一致。                          |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。                          |
