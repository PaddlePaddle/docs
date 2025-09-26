## [paddle 参数更多]torchvision.transforms.functional.to_tensor

### [torchvision.transforms.functional.to_tensor](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.to_tensor.html)

```python
torchvision.transforms.functional.to_tensor(pic: Union[PIL.Image.Image, numpy.ndarray])
```

### [paddle.vision.transforms.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/to_tensor_cn.html)

```python
paddle.vision.transforms.to_tensor(
    pic: Union[PIL.Image.Image, np.ndarray],
    data_format: str = 'CHW'
)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                          |
| --------- | ------------- | ------------------------------ |
| pic       | pic           | 输入图像。                       |
| -         | data_format   | 返回的 Tensor 的格式，PyTorch 无此参数，Paddle 保持默认即可。 |
