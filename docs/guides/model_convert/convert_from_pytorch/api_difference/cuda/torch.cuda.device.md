## [ 输入参数类型不一致 ]torch.cuda.device

### [torch.cuda.device](https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device)

```python
torch.cuda.device(device)
```

### [paddle.device._convert_to_place](https://github.com/PaddlePaddle/Paddle/blob/c8ccc9b154632ef41ade1b8e97b87d54fde7e8f8/python/paddle/device/__init__.py#L174)

```python
paddle.device._convert_to_place(device)
```

其中 PyTorch 与 Paddle 的参数支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                             |
| ------- | ------------ | -------------------------------------------------------------------------------- |
| device  | device           | GPU 的设备 ID, PyTorch 支持 torch.device 和 int，Paddle 支持 str，需要转写。 |

### 转写示例

#### device: 特定的运行设备

```python
# PyTorch 写法
torch.cuda.device('cuda:0')

# Paddle 写法
paddle.device._convert_to_place('gpu:0')

# PyTorch 写法
torch.cuda.device(2)

# Paddle 写法
paddle.device._convert_to_place('gpu:2')

```
