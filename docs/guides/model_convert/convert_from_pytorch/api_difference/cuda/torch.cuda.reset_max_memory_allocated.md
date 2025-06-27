## [参数完全一致]torch.cuda.reset_max_memory_allocated

### [torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.reset_max_memory_allocated.html#torch.cuda.reset_max_memory_allocated)

```python
torch.cuda.reset_max_memory_allocated(device)
```

### [paddle.device.cuda.reset_max_memory_allocated](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/reset_max_memory_allocated_cn.html)

```python
paddle.device.cuda.reset_max_memory_allocated(device)
```

功能不一致（Pytorch 通过调用 reset_peak_memory_stats 函数实现，重置所有 CUDA 内存分配器跟踪的峰值统计。PaddlePaddle 仅重置分配给 Tensor 的显存峰值统计），参数完全一致（PyTorch 参数是 PaddlePaddle 参数子集），具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str。 |
