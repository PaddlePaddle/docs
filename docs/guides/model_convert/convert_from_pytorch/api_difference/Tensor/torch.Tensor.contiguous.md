## torch.Tensor.contiguous
### [torch.Tensor.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html?highlight=contiguous#torch.Tensor.contiguous)

```python
torch.Tensor.contiguous(memory_format)
```

### [paddle]

```python
-
```

该函数用于返回“连续”存储的 tensor ，paddle 无此功能函数，无需此功能：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| memory_format           | -      | 用于返回“连续”存储的 tensor                                     |

### 转写示例

```python
# torch 写法
torch.Tensor.contiguous(memory_format=torch.contiguous_format)

# paddle 写法，无需此功能
-
```
