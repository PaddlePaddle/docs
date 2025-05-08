## [ 输入参数类型不一致 ]torch.cuda.synchronize

### [torch.cuda.synchronize](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html#torch.cuda.synchronize)

```python
torch.cuda.synchronize(device)
```

### [paddle.device.cuda.synchronize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/synchronize_cn.html)

```python
paddle.device.cuda.synchronize(device)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ |-----------------------------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 和 int。 PaddlePaddle 支持 paddle.CUDAPlace、int 、str，需要转写 |

### 转写示例
#### device: 特定的运行设备

```python
# PyTorch 写法
torch.cuda.synchronize('cuda:0')

# Paddle 写法
paddle.device.cuda.synchronize('gpu:0')

# PyTorch 写法
torch.cuda.synchronize(2)

# Paddle 写法
paddle.device.cuda.synchronize('gpu:2')
```
