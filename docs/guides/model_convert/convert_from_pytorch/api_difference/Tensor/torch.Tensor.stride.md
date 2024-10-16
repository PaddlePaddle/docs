## [ torch 参数更多 ] torch.Tensor.stride

### [torch.Tensor.stride](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html#torch-tensor-stride)

```python
torch.Tensor.stride(dim=None)
```

### [paddle.Tensor.get_strides](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tensor)

```python
paddle.Tensor.get_strides()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射


| PyTorch | PaddlePaddle | 备注                                                                                          |
| ------- | ------------ | --------------------------------------------------------------------------------------------- |
| dim     | -            | 返回指定维度的步长， Pytorch 为可选值，默认返回全部步长，此时无需转写，当有输入值时，需要转写。 |

### 转写示例

```python
# torch 版本, 默认返回全部
x.stride(dim=None)

# Paddle 版本
if dim is not None:
    x.get_strides()[dim]
else:
    x.get_strides()
```
