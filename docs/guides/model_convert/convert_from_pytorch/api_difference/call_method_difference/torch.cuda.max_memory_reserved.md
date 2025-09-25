## [参数完全一致]torch.cuda.max_memory_reserved

### [torch.cuda.max_memory_reserved](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved)

```python
torch.cuda.max_memory_reserved(device)
```

### [paddle.device.cuda.max_memory_reserved](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/max_memory_reserved_cn.html)

```python
paddle.device.cuda.max_memory_reserved(device)
```

功能一致，参数完全一致（PyTorch 参数是 PaddlePaddle 参数子集），具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str。 |
