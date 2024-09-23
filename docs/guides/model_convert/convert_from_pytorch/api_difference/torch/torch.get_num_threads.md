## [ 组合替代实现 ]torch.get_num_threads

### [torch.get_num_threads](https://pytorch.org/docs/stable/generated/torch.get_num_threads.html)

```python
torch.get_num_threads()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.get_num_threads()

# Paddle 写法
import multiprocessing
def get_num_threads():
    device = paddle.device.get_device()
    if 'cpu' in device:
        return multiprocessing.cpu_count()

get_num_threads()
```
