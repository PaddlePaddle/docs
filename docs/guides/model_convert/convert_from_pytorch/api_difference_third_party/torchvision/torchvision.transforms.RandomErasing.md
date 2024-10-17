## [paddle 参数更多]torchvision.transforms.RandomErasing

### [torchvision.transforms.RandomErasing](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html?highlight=randomerasing#torchvision.transforms.RandomErasing)

```python
torchvision.transforms.RandomErasing(p = 0.5,
                                     scale = (0.02, 0.33),
                                     ratio = (0.3, 3.3),
                                     value = 0,
                                     inplace = False)
```

### [paddle.vision.transforms.RandomErasing](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomErasing_cn.html)

```python
paddle.vision.transforms.RandomErasing(prob = 0.5,
                                       scale = (0.02, 0.33),
                                       ratio = (0.3, 3.3),
                                       value = 0,
                                       inplace = False,
                                       keys = None)
```

两者功能基本一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision   | PaddlePaddle     | 备注           |
| ------------- | -------------- | --------------- |
| p             | prob          | 输入数据被执行擦除操作的概率，仅参数名不一致。 |
| scale         | scale         | 擦除区域面积在输入图像的中占比范围。 |
| ratio         | ratio         | 擦除区域的纵横比范围。 |
| value         | value         | 擦除区域中像素将被替换为的值。 |
| inplace       | inplace       | 该变换是否在原地操作。                                         |
| -             | keys          | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。     |
