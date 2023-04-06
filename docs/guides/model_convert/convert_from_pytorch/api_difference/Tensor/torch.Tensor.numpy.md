## [ torch 参数更多 ] torch.Tensor.numpy

### [torch.Tensor.numpy](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html?highlight=numpy#torch.Tensor.numpy)

```python
torch.Tensor.numpy(force=False)
```

### [paddle.Tensor.numpy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#numpy)

```python
paddle.Tensor.numpy()
```

两者功能一致，用于将当前 Tensor 转化为 numpy.ndarray。

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| force             | -            | 若force为默认，则返回的ndarray和tensor将共享它们的存储空间，Paddle暂无转写方式。                     |
