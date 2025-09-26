## [ paddle 参数更多 ] torch.ShortTensor

### [torch.ShortTensor](https://pytorch.org/docs/stable/tensors.html)

```python
torch.ShortTensor(data)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data, dtype='int16', place='cpu')
```

Paddle 比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                        |
| ------- | ------------ | ----------------------------------------------------------- |
| -       | dtype        | Tensor 的数据类型，PyTorch 无此参数，Paddle 需设置为 'int16'。   |
| -       | place        | Tensor 的设备，PyTorch 无此参数，Paddle 需设置为 'cpu' 。         |
