## [参数完全一致]torchvision.transforms.functional.erase

### [torchvision.transforms.functional.erase](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.erase.html?highlight=erase#torchvision.transforms.functional.erase)

```python
torchvision.transforms.functional.erase(
    img: Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: Tensor,
    inplace: bool = False
)
```

### [paddle.vision.transforms.erase](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/erase_cn.html)

```python
paddle.vision.transforms.erase(
    img: Union[paddle.Tensor, np.ndarray, PIL.Image.Image],
    i: int,
    j: int,
    h: int,
    w: int,
    v: Union[paddle.Tensor, np.ndarray],
    inplace: bool = False
)
```

两者功能一致，输入参数类型一致。

### 参数映射

| torchvision | PaddlePaddle    | 备注                                                         |
| ----------- | --------------- | ------------------------------------------------------------ |
| img         | img             | 输入图片。 |
| i           | i               | 擦除区域左上角的纵坐标。  |
| j           | j               | 擦除区域左上角的横坐标。  |
| h           | h               | 擦除区域的高度。       |
| w           | w               | 擦除区域的宽度。       |
| v           | v               | 替换擦除区域中像素的值。 |
| inplace     | inplace         | 是否进行原地操作。    |
