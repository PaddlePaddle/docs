## [torch 参数更多]torch.distributed.send

### [torch.distributed.send](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send)

```python
torch.distributed.send(tensor, dst, group=None, tag=0)
```

### [paddle.distributed.send](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/send_cn.html)

```python
paddle.distributed.send(tensor, dst=0, group=None, use_calc_stream=True)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| tensor  | tensor          | 需要发送的 Tensor。                                               |
| dst     | dst             | 接收者的标识符。                                                  |
| group   | group           | new_group 返回的 Group 实例，或者设置为 None 表示默认地全局组。   |
| tag     | -               | 匹配接收标签，Paddle 无此参数，暂无转写方式。                     |
| -       | use_calc_stream | 标识使用计算流还是通信流，PyTorch 无此参数，Paddle 保持默认即可。 |
