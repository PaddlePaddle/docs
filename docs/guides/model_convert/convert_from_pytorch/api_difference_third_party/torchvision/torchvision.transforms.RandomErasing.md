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
                                       inplace False,
                                       keys = None)
```

两者功能基本一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision         | PaddlePaddle     | 备注                                                         |
| --------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| p (float)                                     | prob (float)                               | 参数名称不同，Paddle 使用 `prob` 替代 `p`，表示擦除操作的概率。 |
| scale (Tuple[float, float])                   | scale (Tuple[float, float])                    |  擦除区域面积在输入图像的中占比范围。 |
| ratio (Tuple[float, float])                   | ratio (Tuple[float, float])                    |  擦除区域的纵横比范围。 |
| value (int or float or Tuple[int, int, int] or str) | value (int or float or Tuple[int, int, int] or str)    | 擦除区域中像素将被替换为的值。 |
| inplace (bool)                                | inplace (bool)                             | 该变换是否在原地操作。                                         |
| -                                             | keys (List[str] or Tuple[str], optional)   | Paddle 支持 `keys` 参数，torchvision 不支持。 |
