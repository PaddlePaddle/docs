## [ torch参数更多 ] torch.Tensor.stride

### [torch.Tensor.stride](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html#torch-tensor-stride)

```python
torch.Tensor.stride(dim)
```

### [paddle.Tensor.get_strides](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tensor)

```python
paddle.Tensor.get_strides()

```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                           |
| ------- | ------------ | -------------------------------------------------------------- |
| dim     | -            | 返回指定维度的步长，默认返回全部步长，paddle不支持，需要转写。 |

### 转写示例

```python
# torch 版本, 默认返回全部
x.stride(dim)

# Paddle 版本
x.get_strides() if dim is None else x.get_strides()[dim]
```
