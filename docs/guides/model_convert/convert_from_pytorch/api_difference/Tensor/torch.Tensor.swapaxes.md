## [参数不一致]torch.Tensor.swapaxes

### [torch.Tensor.swapaxes](https://pytorch.org/docs/1.13/generated/torch.Tensor.swapaxes.html)

```python
torch.Tensor.swapaxes(axis0, axis1)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
paddle.Tensor.transpose(perm, name=None)
```

Pytorch 的 ``axis0, axis1`` 与 Paddle 的 ``perm`` 用法不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| axis0, axis1 | perm | torch 的 axis0 与 axis1 表示要交换的两个轴，为整数。 paddle 的 perm 表示重排的维度序列，为 list/tuple 。需要转写。|

### 转写示例
#### axis0, axis1: 表示要交换的两个轴
```python
# pytorch
x = torch.randn(2, 3, 5)
x_swapaxes = x.swapaxes(0,1)

# paddle
x = paddle.randn([2, 3, 5])
x_transposed = x.transpose(perm=[1, 0, 2])
```
