## [ 组合替代实现 ]torch.nn.modules.utils._pair

### [torch.nn.modules.utils._pair](https://github.com/pytorch/pytorch/blob/1f4d4d3b7836d38d936a21665e6b2ab0b39d7092/torch/nn/modules/utils.py#L198)

```python
torch.nn.modules.utils._pair(x)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.nn.modules.utils._pair(x)

# Paddle 写法
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_ntuple(n=2, name="parse")(x)
```
