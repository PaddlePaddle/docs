## [组合替代实现]torch.nn.functional._Reduction.get_enum

### [torch.nn.functional._Reduction.get_enum](https://github.com/pytorch/pytorch/blob/3045b16488f14c9d941d33d63417e6ea52fb2544/torch/nn/_reduction.py#L7)

```python
torch.nn.functional._Reduction.get_enum(reduction)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.nn.functional._Reduction.get_enum(reduction)

# Paddle 写法
def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret

get_enum(reduction)
```
