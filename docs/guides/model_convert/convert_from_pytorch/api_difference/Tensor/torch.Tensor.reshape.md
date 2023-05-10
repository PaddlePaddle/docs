## [参数不一致]torch.Tensor.reshape

### [torch.Tensor.reshape](https://pytorch.org/docs/1.13/generated/torch.Tensor.reshape.html)

```python
torch.Tensor.reshape(*shape)
```

### [paddle.Tensor.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#reshape-shape-name-none)

```python
paddle.Tensor.reshape(shape, name=None)
```

Pytorch 的 `*shape` 参数与 Paddle 的 `shape` 参数用法不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *shape | shape | torch 的 *shape 既可以接收 list 也可接收可变参数。需要转写。|

### 转写示例
#### *shape: 新数组的维度序列，可变参数用法
```python
# pytorch
x = torch.randn(2, 3, 5)
x_reshape = x.reshape(6,5)

# paddle
x = paddle.randn([2, 3, 5])
x_tile_tuple = x.tile((6,5))
```
