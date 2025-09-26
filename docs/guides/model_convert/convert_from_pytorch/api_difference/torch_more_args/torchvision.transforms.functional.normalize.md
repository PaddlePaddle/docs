## [torch 参数更多]torchvision.transforms.functional.normalize

### [torchvision.transforms.functional.normalize](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.normalize.html)

```python
torchvision.transforms.functional.normalize(tensor, mean, std, inplace = False)
```

### [paddle.vision.transforms.normalize](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/normalize_cn.html)

```python
paddle.vision.transforms.normalize(img, mean = 0.0, std = 1.0, data_format = 'CHW', to_rgb = False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                              |
| --------- | ---------- | ---------------------------------------------------- |
| tensor    | img        | 用于归一化的数据，仅参数名不一致。 |
| mean      | mean       | 用于每个通道归一化的均值。                                   |
| std       | std        | 用于每个通道归一化的标准差值。                               |
| inplace   | -          | 是否原地修改，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| -         | data_format| 用于指定数据格式，PyTorch 无此参数，Paddle 保持默认即可。 |
| -         | to_rgb     | 是否将图像转换为 RGB 格式，PyTorch 无此参数，Paddle 保持默认即可。 |
