## [ 组合替代实现 ]torch.set_num_interop_threads

### [torch.set_num_interop_threads](https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html)

```python
torch.set_num_interop_threads()
```

设置 CPU 上用于操作间并行的线程数 (例如，在 JIT 解释器中)，Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.set_num_interop_threads(2)

# Paddle 写法
os.environ['CPU_NUM'] = '2'
```
