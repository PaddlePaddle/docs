## [参数完全一致]torchvision.transforms.functional.erase

### [torchvision.transforms.functional.erase](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.erase.html?highlight=erase#torchvision.transforms.functional.erase)

```python
torchvision.transforms.functional.erase(img: Tensor,
                                        i: int,
                                        j: int,
                                        h: int,
                                        w: int,
                                        v: Tensor,
                                        inplace: bool = False)
```

### [paddle.vision.transforms.erase](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/erase_cn.html)

```python
paddle.vision.transforms.erase(img: paddle.Tensor | np.ndarray | PIL.Image.Image,
                               i: int,
                               j: int,
                               h: int,
                               w: int,
                               v: paddle.Tensor | np.ndarray,
                               inplace: bool = False)
```


### 参数映射

| torchvision.transforms.functional.erase | paddle.vision.transforms.erase      | 备注                                                         |
| --------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| img (Tensor)                            | img (paddle.Tensor \| np.ndarray \| PIL.Image.Image) | 输入类型 torchvision 仅支持 Tensor，Paddle 支持 Tensor、numpy.ndarray 和 PIL.Image。 |
| i (int)                                 | i (int)                             | 擦除区域左上角的纵坐标，参数名称和功能一致。                  |
| j (int)                                 | j (int)                             | 擦除区域左上角的横坐标，参数名称和功能一致。                  |
| h (int)                                 | h (int)                             | 擦除区域的高度，参数名称和功能一致。                         |
| w (int)                                 | w (int)                             | 擦除区域的宽度，参数名称和功能一致。                         |
| v (Tensor)                              | v (paddle.Tensor \| np.ndarray)     | 擦除区域的填充值，torchvision 仅支持 Tensor，Paddle 支持 Tensor 和 numpy.ndarray。当输入为 PIL.Image 时，Paddle 的 `v` 参数需为 numpy.ndarray 类型。 |
| inplace (bool)                          | inplace (bool)                      | 是否进行原地操作，参数名称和功能一致。                       |
