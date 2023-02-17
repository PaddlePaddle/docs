## torch.tensor
### [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html?highlight=from_numpy#torch.from_numpy)

```python
torch.from_numpy(ndarray)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data,
                 dtype=None,
                 place=None,
                 stop_gradient=True)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| ndarray       | data         | 表示需要转换的数据，PyTorch 只能传入 numpy.ndarray，Paddle 可以传入 scalar、list、tuple、numpy.ndarray、paddle.Tensor。 |
| -             | dtype        | 表示数据类型，PyTorch 无此参数，paddle 保持默认即可。               |
| -             | place        | 表示 Tensor 存放位置，PyTorch 无此参数，paddle 需设置为 paddle.CPUPlace()。       |
| -             | stop_gradient| 表示是否阻断梯度传导，PyTorch 无此参数，paddle 保持默认即可。                   |
