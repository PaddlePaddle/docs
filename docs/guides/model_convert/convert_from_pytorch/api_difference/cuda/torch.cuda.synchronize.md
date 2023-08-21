## [参数完全一致]torch.cuda.synchronize

### [torch.cuda.synchronize](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html#torch.cuda.synchronize)

```python
torch.cuda.synchronize(device)
```

### [paddle.device.cuda.synchronize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/synchronize_cn.html)

```python
paddle.device.cuda.synchronize(device)
```

功能一致，参数完全一致（PyTorch 参数是 PaddlePaddle 参数子集），具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str。 |
