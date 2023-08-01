## [torch 参数更多]torch.Tensor.size

### [torch.Tensor.size](https://pytorch.org/docs/stable/generated/torch.Tensor.size.html#torch.Tensor.size)

```python
torch.Tensor.size(dim=None)
```

### [paddle.Tensor.shape](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#shape)

```python
paddle.Tensor.shape
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | ------- |
| dim     | -            | 表示获取大小的轴，Paddle 无此参数，需要转写。 |

### 转写示例

```python
# Pytorch 写法
torch.tensor([-1, -2, 3]).size(0)

# Paddle 写法
paddle.to_tensor([-0.4, -0.2, 0.1, 0.3]).shape[0]
```
