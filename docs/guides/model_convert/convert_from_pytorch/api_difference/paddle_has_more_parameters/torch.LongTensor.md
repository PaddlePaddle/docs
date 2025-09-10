## [ paddle 参数更多 ] torch.LongTensor

### [torch.LongTensor](https://pytorch.org/docs/stable/tensors.html)

```python
torch.LongTensor(data)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data, dtype='int64', place='cpu')
```

Paddle 比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                        |
| ------- | ------------ | ----------------------------------------------------------- |
| data    | data         | 要转换的数据。 |
| -       | dtype        | Tensor 的数据类型，PyTorch 无此参数，Paddle 需设置为 'int64'。   |
| -       | place        | Tensor 的设备，PyTorch 无此参数，Paddle 需设置为 'cpu' 。         |
