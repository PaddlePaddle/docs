## [ 参数默认值不一致 ]torch.set_num_threads

### [torch.set_num_threads](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html)

```python
torch.set_num_threads(int)
```

### [paddle.static.cpu_places](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/cpu_places_cn.html)

```python
paddle.static.cpu_places(device_count=None)
```

其中 PyTorch 和 Paddle 功能一致，参数默认值不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| -       | device_count | 要设置的 Cpu 线程数，仅参数名不一致。 |
