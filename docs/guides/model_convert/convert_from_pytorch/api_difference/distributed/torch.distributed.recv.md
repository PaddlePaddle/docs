## [torch 参数更多]torch.distributed.recv

### [torch.distributed.recv](https://pytorch.org/docs/1.13/distributed.html#torch.distributed.recv)

```python
torch.distributed.recv(tensor, src=None, group=None, tag=0)
```

### [paddle.distributed.recv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/recv_cn.html)

```python
paddle.distributed.recv(tensor, src=0, group=None, use_calc_stream=True)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| tensor  | tensor          | 接收数据的 Tensor。                                               |
| src     | src             | 发送者的标识符。                                                  |
| group   | group           | new_group 返回的 Group 实例，或者设置为 None 表示默认地全局组。   |
| tag     | -               | 匹配接收标签，Paddle 无此参数，暂无转写方式。   |
| -       | use_calc_stream | 标识使用计算流还是通信流，PyTorch 无此参数，Paddle 保持默认即可。 |
