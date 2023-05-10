## [参数不一致]torch.Tensor.swapdims

### [torch.Tensor.swapdims](https://pytorch.org/docs/1.13/generated/torch.Tensor.swapdims.html)

```python
torch.Tensor.swapdims(dim0, dim1)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
paddle.Tensor.transpose(perm, name=None)
```

Pytorch 的 `dim0, dim1` 与 Paddle 的 `perm` 用法不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim0, dim1 | perm | torch 的 dim0 与 dim1 表示要交换的两个维度, 为整数。 paddle 的 perm 表示重排的维度序列，为 list/tuple 。需要转写。|

### 转写示例
#### dim0, dim1: 表示要交换的两个维度
```python
# pytorch
x = torch.randn(2, 3, 5)
x_swapdims = x.swapdims(0,1)

# paddle
x = paddle.randn([2, 3, 5])
x_transposed = x.transpose(perm=[1, 0, 2])
```
