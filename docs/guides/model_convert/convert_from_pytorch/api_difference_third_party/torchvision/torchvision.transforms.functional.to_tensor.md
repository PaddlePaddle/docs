## [paddle 参数更多]torchvision.transforms.functional.to_tensor

### [torchvision.transforms.functional.to_tensor](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.to_tensor.html)

```python
torchvision.transforms.functional.to_tensor(pic: Union[PIL.Image.Image, numpy.ndarray])
```

### [paddle.vision.transforms.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/to_tensor_cn.html#to-tensor)

```python
paddle.vision.transforms.to_tensor(pic: PIL.Image.Image | numpy.ndarray, data_format: str = 'CHW')
```

两者功能基本一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision.transforms.functional.to_tensor | paddle.vision.transforms.to_tensor | 备注                                               |
| ------------------------------------------- | ----------------------------------- | -------------------------------------------------- |
| pic (PIL.Image.Image \| numpy.ndarray)      | pic (PIL.Image.Image \| numpy.ndarray) | 输入图像，参数类型和名称一致。                       |
| -                                           | data_format (str，可选，默认值='CHW') | Paddle 特有参数，指定返回的 Tensor 数据格式，可选 `'CHW'` 或 `'HWC'`。 |
