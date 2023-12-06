## [ 仅参数名不一致 ]torch.from_numpy
### [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html?highlight=from_numpy#torch.from_numpy)

```python
torch.from_numpy(ndarray)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data,
                 dtype=None,
                 place=None,
                 stop_gradient=True)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> ndarray </font>      | <font color='red'> data </font>  | 表示需要转换的数据， PyTorch 只能传入 numpy.ndarray ， Paddle 可以传入 scalar 、 list 、 tuple 、 numpy.ndarray 、 paddle.Tensor 。 |
| -             | <font color='red'> dtype  </font>   | 表示数据类型， PyTorch 无此参数， Paddle 保持默认即可。               |
| -             | <font color='red'> place </font>       | 表示 Tensor 存放位置， PyTorch 无此参数， Paddle 需设置为 paddle.CPUPlace()。   |
| -             | <font color='red'> stop_gradient </font> | 表示是否阻断梯度传导， PyTorch 无此参数， Paddle 保持默认即可。                   |
