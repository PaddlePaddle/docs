## [输入参数类型不一致]torch.Tensor.cuda

### [torch.Tensor.cuda](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda)

```python
torch.Tensor.cuda(device=None, non_blocking=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cuda](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cuda-device-id-none-blocking-false)

```python
paddle.Tensor.cuda(device_id=None, blocking=False)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                     |
| ------------- | ------------ | ------------------------------------------------------------------------ |
| device        | device_id    | 目标 GPU 设备，输入参数类型不一致，需要转写。                                          |
| non_blocking  | blocking     | 是否同步或异步拷贝，PyTorch 和 Paddle 取值相反，需要转写。                                           |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |


### 转写示例

#### non_blocking: 同步或异步拷贝

```python
# PyTorch 写法
tensor.cuda(non_blocking=True)

# Paddle 写法
tensor.cuda(blocking=False)
```

#### device: 目标 GPU 设备

```python
# PyTorch 写法
tensor.cuda("cuda:0")

# Paddle 写法
tensor.cuda(0)

# PyTorch 写法
tensor.cuda(0)

# Paddle 写法
tensor.cuda(0)
```
