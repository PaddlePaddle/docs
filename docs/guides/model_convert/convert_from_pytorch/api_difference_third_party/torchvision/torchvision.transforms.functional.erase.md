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


### 参数映射

| torchvision | PaddlePaddle     | 备注                                                         |
| --------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| img                             | img  | Paddle 支持更多类型，其中包含 PyTorch 支持的类型。 |
| i                                  | i                              | 擦除区域左上角的纵坐标，参数名称和功能一致。                  |
| j                                  | j                              | 擦除区域左上角的横坐标，参数名称和功能一致。                  |
| h                                  | h                              | 擦除区域的高度，参数名称和功能一致。                         |
| w                                  | w                              | 擦除区域的宽度，参数名称和功能一致。                         |
| v                               | v      | Paddle 支持更多类型，其中包含 PyTorch 支持的类型。 |
| inplace                           | inplace                       | 是否进行原地操作。                       |
