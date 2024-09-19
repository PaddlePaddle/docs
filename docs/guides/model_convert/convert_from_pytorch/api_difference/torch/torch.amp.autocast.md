### [ torch 参数更多 ] torch.amp.autocast

### [torch.amp.autocast](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)

```python
torch.amp.autocast(device_type,
                   dtype=None,
                   enabled=True,
                   cache_enabled=None)
```

### [paddle.amp.auto_cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html#auto-cast)

```python
paddle.amp.auto_cast(enable=True,
                     custom_white_list=None,
                     custom_black_list=None,
                     level='O1',
                     dtype='float16',
                     use_promote=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | Paddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| device_type         | -      | 指定设备类型,Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                         |
| dtype           | dtype      | 指定自动混合精度的计算类型         |
| enabled         | enable  | 是否启用自动混合精度。 |
| cache_enabled        | -      | 启用或禁用 CUDA 图形缓存 Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -        | custom_white_list      | 白名单，通常不需要设置。 |
| -        | custom_black_list      | 黑名单，通常不需要设置。 |
| -        | level      | 混合精度训练的优化级别，可为 O1 、O2 或者 OD 模式。 |
| -        | use_promote      | 当一个算子存在 float32 类型的输入时，按照 Promote to the Widest 原则，选择 float32 数据类型进行计算 |