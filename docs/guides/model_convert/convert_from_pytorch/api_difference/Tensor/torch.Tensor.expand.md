## torch.Tensor.expand
### [torch.Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html?highlight=expand#torch.Tensor.expand)

```python
torch.Tensor.expand(*size)
```

### [paddle.Tensor.expand](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#expand-shape-name-none)

```python
paddle.Tensor.expand(shape, name=None)
```

两者功能一致，仅参数名不一致，具体差异如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 扩张后的维度                                             |

### 转写示例

```python
# torch 写法
torch.Tensor.expand(3, 4)

# paddle 写法
paddle.Tensor.expand(shape=[3, 4])
```
