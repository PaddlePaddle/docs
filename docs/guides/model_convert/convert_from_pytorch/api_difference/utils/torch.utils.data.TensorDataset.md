## [参数用法不一致]torch.utils.data.TensorDataset

### [torch.utils.data.TensorDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset)

```python
torch.utils.data.TensorDataset(tensor1, tensor2, tensor3)
```

### [paddle.io.TensorDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/TensorDataset_cn.html)

```python
paddle.io.TensorDataset([tensor1, tensor2, tensor3])
```

paddle 参数和 torch 参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                             |
|----------|--------------|------------------------------------------------|
| *tensors | tensors      | 输入的 Tensor， PyTorch 是多个元素， Paddle 是列表或元组，需要进行转写 |

### 转写示例

#### tensors：输入的 Tensor

```python
# Pytorch 写法
torch.utils.data.TensorDataset(tensor1, tensor2, tensor3)

# Paddle 写法
paddle.io.TensorDataset([tensor1, tensor2, tensor3])
```
