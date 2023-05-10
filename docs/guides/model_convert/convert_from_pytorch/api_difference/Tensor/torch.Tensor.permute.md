## [参数不一致]torch.Tensor.permute

### [torch.Tensor.permute](https://pytorch.org/docs/1.13/generated/torch.Tensor.permute.html)

```python
torch.Tensor.permute(*dims)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
paddle.Tensor.transpose(perm, name=None)
```

torch 的 `*dims` 与 paddle 的 `perm` 两者部分参数用法不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *dims | perm | torch 支持可变参数或 list/tuple，paddle 仅支持 list/tuple，参数用法不一致，需要进行转写。 |

### 转写示例
#### *dim: Tensor 的维度序列，可变参数用法
```python
# pytorch
x = torch.randn(2, 3, 5)
x_permuted_dim = x.permute(2,0,1)

# paddle
x = paddle.randn([2, 3, 5])
x_transposed_list = x.transpose([2, 0, 1])
```
