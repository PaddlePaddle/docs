## [ 参数不一致 ]torch.Tensor.permute

### [torch.Tensor.permute](https://pytorch.org/docs/stable/generated/torch.Tensor.permute.html)

```python
torch.Tensor.permute(*dims)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
paddle.Tensor.transpose(perm, name=None)
```

Pytorch 的 `*dims` 相比于 paddle 的 `perm` 额外支持可变参数的用法，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *dims  | perm | torch 支持可变参数或 list/tuple，paddle 仅支持 list/tuple。对于可变参数的用法，需要转写。 |

### 转写示例
#### *dims: 可变参数用法
```python
# pytorch
x = torch.randn(2, 3, 5)
y = x.permute(2, 0, 1)

# paddle
x = paddle.randn([2, 3, 5])
y = x.transpose([2, 0, 1])
```
