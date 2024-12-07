## [参数完全一致]torch.cuda.memory_stats

### [torch.cuda.memory_stats](https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats)

```python
torch.cuda.memory_stats(device)
```

### [paddle.device.cuda.memory_stats](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/memory_stats_cn.html)

```python
paddle.device.cuda.memory_stats(device)
```

功能一致（均返回包含CUDA内存分配器统计信息的字典，但字典的具体内容有区别），参数完全一致（PyTorch 参数是 PaddlePaddle 参数子集），具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str。 |
