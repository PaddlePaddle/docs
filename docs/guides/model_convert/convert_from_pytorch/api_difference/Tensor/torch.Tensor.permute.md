## [部分参数不一致]torch.Tensor.permute

### [torch.Tensor.permute](https://pytorch.org/docs/1.13/generated/torch.Tensor.permute.html)

```python
    torch.Tensor.permute(*dims)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
    paddle.Tensor.transpose(perm, name=None)
```

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *dims | perm | dim：torch可以为 *dim和list/tuple的用法; perm：paddle仅为list/tuple的用法 |

### 代码转写

```python
    # pytorch
    x = torch.randn(2, 3, 5)
    x_permuted_dim = x.permute(2,0,1)
    x_permuted_list = x.permute([2,0,1])
    x_permuted_tuple = x.permute((2,0,1))

    # paddle
    x = paddle.randn([2, 3, 5])
    x_transposed_list = x.transpose(perm=[2, 0, 1])
    x_transposed_tuple = x.transpose(perm=(2, 0, 1))
```
