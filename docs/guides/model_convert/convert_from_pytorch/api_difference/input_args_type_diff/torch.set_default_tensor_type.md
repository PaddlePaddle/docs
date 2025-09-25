## [ 输入参数类型不一致 ]torch.set_default_tensor_type

### [torch.set\_default\_tensor\_type](https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html)

```python
torch.set_default_tensor_type(t)
```

### [paddle.set\_default\_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html#set-default-dtype)

```python
paddle.set_default_dtype(d)
```

两者功能一致但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| t       | d            | 浮点张量类型或其名称。PyTorch 支持 `torch.FloatTensor` 或 "torch.FloatTensor"，Paddle 只支持 `dtype`（如 `paddle.float32`），需要转写。 |

### 转写示例

#### t 浮点张量类型或其名称

```python
# PyTorch
torch.set_default_tensor_type(torch.FloatTensor)

# Paddle
paddle.set_default_dtype(paddle.float32)
```
