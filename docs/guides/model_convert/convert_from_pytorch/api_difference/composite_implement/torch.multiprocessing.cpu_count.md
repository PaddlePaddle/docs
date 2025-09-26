## [ 组合替代实现 ]torch.multiprocessing.cpu_count

### [torch.multiprocessing.cpu_count](https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/__init__.py)
```python
torch.multiprocessing.cpu_count()
```

获取系统上可用的 CPU 核心数。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例

```python
# PyTorch 写法
torch.multiprocessing.cpu_count()

# Paddle 写法
import os
os.cpu_count()
```
