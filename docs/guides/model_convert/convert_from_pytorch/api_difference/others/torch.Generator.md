## [ 组合替代实现 ] torch.Generator

### [torch.Generator](https://pytorch.org/docs/2.0/generated/torch.Generator.html#generator)

```python
torch.Generator(device='cpu')
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API转写。

### 转写示例
```python
# torch 写法
g_cpu = torch.Generator()
g_cuda = torch.Generator(device='cuda')

# paddle 写法
g_cpu = paddle.fluid.core.default_cpu_generator()
device = paddle.device.get_device()
g_cuda = paddle.fluid.core.default_cuda_generator(device[-1])
```