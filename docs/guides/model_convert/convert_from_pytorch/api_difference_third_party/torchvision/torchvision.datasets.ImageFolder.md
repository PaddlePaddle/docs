## [torch 参数更多]torchvision.datasets.ImageFolder

### [torchvision.datasets.ImageFolder](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/ImageFolder_cn.html)

```python
torchvision.datasets.ImageFolder(root: Union[str, pathlib.Path], 
                                 transform: Optional[Callable] = None, 
                                 target_transform: Optional[Callable] = None, 
                                 loader: Callable[[str], Any] = default_loader, 
                                 is_valid_file: Optional[Callable[[str], bool]] = None, 
                                 allow_empty: bool = False)
```

### [paddle.vision.datasets.ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)

```python
paddle.vision.datasets.ImageFolder(root: str, 
                                   loader: Optional[Callable] = None, 
                                   extensions: Optional[Union[list[str], tuple[str]]] = None, 
                                   transform: Optional[Callable] = None, 
                                   is_valid_file: Optional[Callable[[str], bool]] = None)
```


### 参数映射

| torchvision | PaddlePaddle | 备注                                      |
| -------------------------------- | ---------------------------------- | ----------------------------------------- |
| root                             | root                               | 根目录路径。                                |
| transform                        | transform                          | 图片数据的预处理，若为 None 即为不做预处理。默认值为 None。 |
| target_transform                 | -                                  | 目标标签的预处理，Paddle 无此参数。         |
| loader                           | loader                             | 图片加载函数。                              |
| is_valid_file                    | is_valid_file                      | 根据每条数据的路径来判断是否合法的一个函数。extensions 和 is_valid_file 不可以同时设置。                  |
| allow_empty                      | -                                  | 是否允许空文件夹，Paddle 无此参数。一般情况下对使用无显著影响，可直接删除。 |
| -                                | extensions                         | Paddle 中支持设定允许的文件后缀，用于限制数据集文件的格式。|

### 转写示例

```python
# PyTorch 写法
dataset = torchvision.datasets.ImageFolder(root="data/dataset",
                                           transform=transform,
                                           loader=custom_loader,
                                           is_valid_file=validate_image)

# Paddle 写法
dataset = paddle.vision.datasets.ImageFolder(root="data/dataset",
                                             transform=transform,
                                             loader=custom_loader,
                                             is_valid_file=validate_image)
```