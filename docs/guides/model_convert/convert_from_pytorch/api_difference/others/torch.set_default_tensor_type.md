## [ 输入参数类型不一致 ] torch.set_default_tensor_type

### [torch.set_default_tensor_type](https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html#torch-set-default-tensor-type)

```python
torch.set_default_tensor_type(d)
```

### [paddle.set_default_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html)

```python
paddle.set_default_dtype(d)
```

两者功能一致，支持的参数类型相同，但输入参数类型不一致，需将 d 转换为 paddle 可识别类型，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | -------------------------------------------------------------------------------------- |
| d           | d            | 全局默认数据类型，均支持所有浮点类型|

### 转写示例
```python
# pytorch
torch.set_default_tensor_type(torch.HalfTensor)
torch.set_default_tensor_type('torch.HalfTensor')
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type('torch.DoubleTensor')

# paddle
paddle.set_default_dtype('float16')
paddle.set_default_dtype('float16')
paddle.set_default_dtype('float32')
paddle.set_default_dtype('float32')
paddle.set_default_dtype('float64')
paddle.set_default_dtype('float64')
```
