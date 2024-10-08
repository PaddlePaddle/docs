## [仅参数名不一致]torch.isin

### [torch.isin](https://pytorch.org/docs/stable/generated/torch.isin.html#torch.isin)

```python
torch.isin(elements, test_elements, *, assume_unique=False, invert=False)
```

### [paddle.isin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/isin_cn.html)

```python
paddle.isin(x, test_x, assume_unique=False, invert=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| elements      | x            | 表示输入的 Tensor，仅参数名不一致。                    |
| test_elements | test_x       | 表示用于检验的 Tensor ，仅参数名不一致。               |
| assume_unique | assum_unique | 表示输入的 Tensor 和用于检验的 Tensor 的元素是否唯一。 |
| invert        | invert       | 表示是否输出反转的结果。                               |
