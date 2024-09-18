## [ paddle 参数更多 ] torch.Tensor.int_repr

### [torch.Tensor.int_repr](https://pytorch.org/docs/stable/generated/torch.Tensor.int_repr.html#torch.Tensor.int_repr)

```python
torch.Tensor.int_repr()
```

### [paddle.tensor.create_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#create-tensor-dtype-name-none-persistable-false)

```python
paddle.tensor.create_tensor(dtype, name=None, persistable=False)

```

Paddle 比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                        |
| ------- | ------------ | ----------------------------------------------------------- |
| -       | dtype        | Tensor 的数据类型，PyTorch 无此参数，Paddle 需设置为 'uint8'。   |
