## torch.Tensor.is_contiguous
### [torch.Tensor.is_contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html?highlight=is_contiguous#torch.Tensor.is_contiguous)

```python
torch.Tensor.is_contiguous(memory_format)
```

### [paddle]

```python
-
```

该函数用于判断 tensor 是否连续，paddle 无此功能函数，无需判断 tensor 是否连续：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| memory_format           | -      | 用于判断 tensor 是否连续                                     |

### 转写示例

```python
# torch 写法
torch.Tensor.is_contiguous(memory_format=torch.contiguous_format)

# paddle 写法，无需判断
-
```
