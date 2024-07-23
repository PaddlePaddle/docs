## [组合替代实现]torch.Generator

### [torch.Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html#generator)

```python
torch.Generator(device='cpu')
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.Generator()

# Paddle 写法
paddle.framework.core.default_cpu_generator()
```

```python
# PyTorch 写法
torch.Generator(device="cuda")

# Paddle 写法
device = paddle.device.get_device()
paddle.framework.core.default_cuda_generator(int(device[-1]))
```
