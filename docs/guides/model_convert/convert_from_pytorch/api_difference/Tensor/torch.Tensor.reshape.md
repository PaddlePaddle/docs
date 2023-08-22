## [ 参数不一致 ]torch.Tensor.reshape

### [torch.Tensor.reshape](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html)

```python
torch.Tensor.reshape(*shape)
```

### [paddle.Tensor.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#reshape-shape-name-none)

```python
paddle.Tensor.reshape(shape, name=None)
```

Pytorch 的 `*shape` 相比于 Paddle 的 `shape` 额外支持可变参数的用法，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *shape | shape | torch 支持可变参数或 list/tuple，paddle 仅支持 list/tuple。对于可变参数的用法，需要转写。 |

### 转写示例
#### *shape: 新数组的维度序列，可变参数用法
```python
# pytorch
x = torch.randn(2, 3, 5)
y = x.reshape(6, 5)

# paddle
x = paddle.randn([2, 3, 5])
y = x.reshape((6, 5))
```
