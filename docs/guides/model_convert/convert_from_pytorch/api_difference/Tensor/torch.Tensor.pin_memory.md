## [ 无参数 ] torch.Tensor.pin_memory

### [torch.Tensor.pin_memory](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html?highlight=pin_mem#torch.Tensor.pin_memory)

```python
torch.Tensor.pin_memory()
```

### [paddle.Tensor.pin_memory](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#pin-memory-y-name-none)

```python
paddle.Tensor.pin_memory()
```



两者功能一致，均无参数，用于将当前 Tensor 的拷贝到固定内存上，且返回的 Tensor 不保留在原计算图中。
