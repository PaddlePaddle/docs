## [paddle 参数更多]torch.cuda.amp.GradScaler

### [torch.cuda.amp.GradScaler](https://pytorch.org/docs/1.13/amp.html#torch.cuda.amp.GradScaler)

```python
torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
```

### [paddle.amp.GradScaler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/GradScaler_cn.html)

```python
paddle.amp.GradScaler(enable=True, init_loss_scaling=32768.0, incr_ratio=2.0, decr_ratio=0.5, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, use_dynamic_loss_scaling=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch         | PaddlePaddle             | 备注                                                         |
| --------------- | ------------------------ | ------------------------------------------------------------ |
| init_scale      | init_loss_scaling        | 初始 loss scaling 因子。与 Pytorch 默认值不同，需要转写。                                  |
| growth_factor   | incr_ratio               | 增大 loss scaling 时使用的乘数。                             |
| backoff_factor  | decr_ratio               | 减小 loss scaling 时使用的小于 1 的乘数。                    |
| growth_interval | incr_every_n_steps       | 连续 n 个 steps 的梯度都是有限值时，增加 loss scaling。与 Pytorch 默认值不同，需要转写。     |
| enabled         | enable                  | 是否使用 loss scaling。                                      |
| -               | decr_every_n_nan_or_inf  | 累计出现 n 个 steps 的梯度为 nan 或者 inf 时，减小 loss scaling，PyTorch 无此参数，Paddle 保持默认即可。 |
| -               | use_dynamic_loss_scaling | 是否使用动态的 loss scaling，PyTorch 无此参数，Paddle 保持默认即可。 |


### 转写示例
#### init_scale：表示初始化 loss scaling 因子
```python
# Pytorch 写法
scale = torch.cuda.amp.GradScaler(init_scale=65536.0)

# Paddle 写法
scale = torch.cuda.amp.GradScaler(init_loss_scaling=65536.0)
```
#### growth_interval：连续 n 个 steps 的梯度都是有限值时
```python
# Pytorch 写法
scale = torch.cuda.amp.GradScaler(growth_interval=2000)

# Paddle 写法
scale = torch.cuda.amp.GradScaler(incr_every_n_steps=2000)
```
