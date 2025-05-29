## [ 组合替代实现 ]torch.cuda.set_per_process_memory_fraction

### [torch.cuda.set_per_process_memory_fraction](https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html)

```python
torch.cuda.set_per_process_memory_fraction(fraction, device=None)
```

限制当前进程在指定 GPU 上最多能分配的显存比例，Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.cuda.set_per_process_memory_fraction(0.5)

# Paddle 写法
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'
```
