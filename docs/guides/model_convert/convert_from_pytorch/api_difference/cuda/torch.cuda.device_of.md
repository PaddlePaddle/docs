## [ 输入参数类型不一致 ]torch.cuda.device_of

### [torch.cuda.device_of](https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html)

```python
torch.cuda.device_of(obj)
```

### [paddle.static.device_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/device_guard_cn.html#device-guard)

```python
paddle.static.device_guard(device=None)
```

其中 PyTorch 与 Paddle 的参数支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                             |
| ------- | ------------ | -------------------------------------------------------------------------------- |
| obj  | device          | PyTorch 输入为在所选设备上分配的对象，Paddle 为 指定上下文中使用的设备，需要转写。 |


### 转写示例
```python

# PyTorch 写法
with torch.cuda.device_of(innput_tensor)

# Paddle 写法
with paddle.static.device_guard("gpu:{}".format(input_tensor.place.gpu_device_id()))
