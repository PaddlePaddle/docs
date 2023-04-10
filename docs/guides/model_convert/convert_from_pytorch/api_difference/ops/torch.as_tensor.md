## [ 仅参数名不一致 ]torch.as_tensor
### [torch.as_tensor](https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor)

```python
torch.as_tensor(data,
                dtype=None,
                device=None)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data,
                dtype=None,
                place=None,
                stop_gradient=True)
```

两者功能一致，性能某些用法下比 PyTorch 差（比如如果输入一个 Tensor ， Pytorch 会直接返回， Paddle 会复制后返回）。此外， Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| data          | data         | 表示输入的 Tensor 。                                     |
| dtype           | dtype            | 表示 Tensor 的数据类型。               |
| <font color='red'> device </font>           | <font color='red'> place </font>            | 表示 Tensor 的存放位置，仅参数名不同。               |
| -           | <font color='red'> stop_gradient </font>            | 表示是否阻断梯度传导， PyTorch 无此参数， Paddle 保持默认即可。             |
