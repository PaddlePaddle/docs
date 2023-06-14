## [paddle 参数更多]torch.cuda.amp.autocast

### [torch.cuda.amp.autocast](https://pytorch.org/docs/1.13/amp.html#torch.cuda.amp.autocast)

```python
torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True)
```

### [paddle.amp.auto_cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/auto_cast_cn.html)

```python
paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O1', dtype='float16')
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle      | 备注                                                         |
| ------------- | ----------------- | ------------------------------------------------------------ |
| enabled       | enabled           | 是否开启自动混合精度。                                       |
| dtype         | dtype             | 使用的数据类型。                                             |
| cache_enabled | -                 | 是否启用权重缓存，Paddle 无此参数，暂无转写方式。 |
| -             | custom_white_list | 自定义算子白名单，PyTorch 无此参数，Paddle 保持默认即可。    |
| -             | custom_black_list | 自定义算子黑名单，PyTorch 无此参数，Paddle 保持默认即可。    |
| -             | level             | 混合精度训练模式，PyTorch 无此参数，Paddle 保持默认即可。    |
