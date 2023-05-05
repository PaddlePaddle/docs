## [部分参数不一致]torch.Tensor.swapdims

### [torch.Tensor.swapdims](https://pytorch.org/docs/1.13/generated/torch.Tensor.swapdims.html)

```python
    torch.Tensor.swapdims(dim0, dim1) 
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
    paddle.Tensor.transpose(perm, name=None)
```

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim0, dim1 | perm | torch的参数为dim0, dim1, 为整数。paddle的perm为list/tuple |

### 代码转写

```python
    # pytorch
    x = torch.randn(2, 3, 5)
    x_swapdims = x.swapdims(0,1)

    # paddle
    x = paddle.randn([2, 3, 5])
    x_transposed = x.transpose(perm=[1, 0, 2])
```