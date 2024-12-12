## [参数完全一致]torch.cuda.reset_peak_memory_stats

### [torch.cuda.reset_peak_memory_stats](https://pytorch.org/docs/stable/generated/torch.cuda.reset_peak_memory_stats.html#torch.cuda.reset_peak_memory_stats)

```python
torch.cuda.reset_peak_memory_stats(device)
```

### [paddle.device.cuda.reset_peak_memory_stats](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/reset_peak_memory_stats_cn.html)

```python
paddle.device.cuda.reset_peak_memory_stats(device)
```

功能一致，参数完全一致（PyTorch 参数是 PaddlePaddle 参数子集），具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str。 |
