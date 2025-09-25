## [参数完全一致]torch.cuda.memory_allocated

### [torch.cuda.memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html#torch.cuda.memory_allocated)

```python
torch.cuda.memory_allocated(device)
```

### [paddle.device.cuda.memory_allocated](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/memory_allocated_cn.html)

```python
paddle.device.cuda.memory_allocated(device)
```

功能一致，参数完全一致（PyTorch 参数是 PaddlePaddle 参数子集），具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str。 |
