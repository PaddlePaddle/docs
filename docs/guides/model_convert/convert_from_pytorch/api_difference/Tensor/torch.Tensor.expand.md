## [ 仅参数名不一致 ]torch.Tensor.expand
### [torch.Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html?highlight=expand#torch.Tensor.expand)

```python
torch.Tensor.expand(*size)
```

### [paddle.Tensor.expand](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#expand-shape-name-none)

```python
paddle.Tensor.expand(shape, name=None)
```

两者功能一致，仅参数名不一致，具体差异如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 扩张后的维度，size 是可变参数，paddle 是 list/tuple           |

### 转写示例
#### size: 扩张后的维度
```python
# torch 写法
x.expand(3, 4)

# paddle 写法
x.expand(shape=[3, 4])
```
