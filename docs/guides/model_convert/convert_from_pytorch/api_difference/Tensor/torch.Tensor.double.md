## [ torch 参数更多 ] torch.Tensor.double

### [torch.Tensor.double](https://pytorch.org/docs/stable/generated/torch.Tensor.double.html#torch-Tensor-double)

```python
torch.Tensor.double(memory_format=torch.preserve_format)
```

### [paddle.Tensor.astype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#astype-dtype)

```python
paddle.Tensor.astype('float64')
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |

### 转写示例

```python
# torch 写法
x.double()

# paddle 写法
x.astype('float64')
```
