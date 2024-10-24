## [输入参数用法不一致]torchvision.datasets.VOCDetection

### [torchvision.datasets.VOCDetection](https://pytorch.org/vision/main/generated/torchvision.datasets.VOCDetection.html)

```python
torchvision.datasets.VOCDetection(root: Union[str, Path], year: str = '2012', image_set: str = 'train', download: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None)
```

### [paddle.vision.datasets.VOC2012](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/VOC2012_cn.html)

```python
paddle.vision.datasets.VOC2012(data_file: Optional[str] = None, mode: str = 'train', transform: Optional[Callable] = None, download: bool = True, backend: Optional[str] = None)
```

指定数据集文件路径的参数 `root` 与指定训练集的参数 `image_set` 的用法不一致，Paddle 只支持 2012 年数据集，但 PyTorch 支持 2007 和 2012 年数据集，具体如下：

### 参数映射

| torchvision        | PaddlePaddle           | 备注                                                       |
| ---------------------- | --------------------- | ---------------------------------------------------------- |
| root                   | data_file             | 数据集文件路径，Paddle 参数 data_file 需含完整的文件名，如 PyTorch 参数 `./data`，对应 Paddle 参数 `./data/voc2012/VOCtrainval_11-May-2012.tar`，需要转写。         |
| year                   | -                     | 数据集年份，Paddle 只支持 2012 年数据集，但 PyTorch 支持 2007 和 2012 年数据集，Paddle 无此参数，暂无转写方式。  |
| image_set              | mode                  | 数据集子集，PyTorch 参数 image_set 可以选择 'train'、'trainval' 或 'val'，而 Paddle 参数 mode 只能为 'train'、'test'，暂无转写方式。 |
| transform              | transform             | 图片数据的预处理。           |
| target_transform       | -                     | 接受目标数据并转换，Paddle 无此参数，暂无转写方式。    |
| download               | download              | 是否自动下载数据集文件。 |
| -                      | backend               | 指定图像类型，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例
#### root：数据集文件路径
```python
# PyTorch 写法
train_dataset = torchvision.datasets.VOCDetection(root='./data', image_set='train')

# Paddle 写法
train_dataset = paddle.vision.datasets.VOC2012(data_file='./data/voc2012/VOCtrainval_11-May-2012.tar', mode='train')
```

#### image_set: 数据集子集
```python
# PyTorch 写法
train_dataset = torchvision.datasets.VOCDetection(root='./data', image_set='train', download=True)

# Paddle 写法
train_dataset = paddle.vision.datasets.VOC2012(data_file='./data/voc2012/VOCtrainval_11-May-2012.tar', mode='train', download=True)
```
