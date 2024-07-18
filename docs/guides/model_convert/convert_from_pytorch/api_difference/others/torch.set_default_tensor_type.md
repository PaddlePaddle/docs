## [ 输入参数类型不一致 ] torch.set_default_tensor_type

### [torch.set_default_tensor_type](https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html#torch-set-default-tensor-type)

```python
torch.set_default_tensor_type(d)
```

### [paddle.set_default_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html)

```python
paddle.set_default_dtype(d)
```

两者功能一致但但输入参数类型不一致，torch 支持浮点张量类型或其名称，paddle 仅支持 dtype，需要转写，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                 |
| ----------- | ------------ | ------------------- |
| d           | d            | 浮点张量类型或其名称.   |

### 转写示例
#### d: 浮点张量类型或其名称
```python
# pytorch 写法
torch.set_default_tensor_type(torch.HalfTensor)
torch.set_default_tensor_type('torch.HalfTensor')
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type('torch.DoubleTensor')

# paddle 写法
paddle.set_default_dtype('float16')
paddle.set_default_dtype('float16')
paddle.set_default_dtype('float32')
paddle.set_default_dtype('float32')
paddle.set_default_dtype('float64')
paddle.set_default_dtype('float64')
```
