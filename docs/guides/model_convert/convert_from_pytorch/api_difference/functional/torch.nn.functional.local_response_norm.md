## [ 仅参数名不一致 ]torch.nn.functional.local_response_norm

### [torch.nn.functional.local_response_norm](https://pytorch.org/docs/stable/generated/torch.nn.functional.local_response_norm.html?highlight=local_response_norm#torch.nn.functional.local_response_norm)

```python
torch.nn.functional.local_response_norm(input,
                                        size,
                                        alpha=0.0001,
                                        beta=0.75,
                                        k=1.0)
```

### [paddle.nn.functional.local_response_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/local_response_norm_cn.html)

```python
paddle.nn.functional.local_response_norm(x,
                                         size,
                                         alpha=1e-4,
                                         beta=0.75,
                                         k=1.,
                                         data_format='NCHW',
                                         name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor ，仅参数名不一致。                                     |
| size          | size         | 表示累加的通道数 。                                     |
| alpha          | alpha         | 表示缩放参数 。                                     |
| beta          | beta         | 表示指数 。                                     |
| k           | k            | 表示位移。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
