## [ 组合替代实现 ]torch.set_num_threads

### [torch.set_num_threads](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html)

```python
torch.set_num_threads(int)
```

设置用于 CPU 上的内部操作并行性的线程数，Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.set_num_threads(2)

# Paddle 写法
os.environ['CPU_NUM'] = '2'
```
