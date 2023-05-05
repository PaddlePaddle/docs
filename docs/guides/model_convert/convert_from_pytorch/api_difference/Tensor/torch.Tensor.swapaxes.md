## [部分参数不一致]torch.Tensor.swapaxes

### [torch.Tensor.swapaxes](https://pytorch.org/docs/1.13/generated/torch.Tensor.swapaxes.html)

```python
    torch.Tensor.swapaxes(axis0, axis1) 
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
    paddle.Tensor.transpose(perm, name=None)
```

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| axis0, axis1 | perm | torch的参数为axis0, axis1, 为整数。paddle的perm为list/tuple |

### 代码转写

```python
    # pytorch
    x = torch.randn(2, 3, 5)
    x_swapaxes = x.swapaxes(0,1)

    # paddle
    x = paddle.randn([2, 3, 5])
    x_transposed = x.transpose(perm=[1, 0, 2])
```