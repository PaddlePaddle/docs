## [仅参数名不一致]torchvision.io.read_file

### [torchvision.io.read_file](https://pytorch.org/vision/main/generated/torchvision.io.read_file.html)

```python
torchvision.io.read_file(path)
```

### [paddle.vision.ops.read_file](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/read_file_cn.html#cn-api-paddle-vision-ops-read-file)

```python
paddle.vision.ops.read_file(filename, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| torchvision        | PaddlePaddle         | 备注                                                       |
| ------------------ | -------------------- | ---------------------------------------------------------- |
| path               | filename             | 文件路径，仅参数名不一致。          |
