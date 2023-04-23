## [ 参数完全一致 ]torch.utils.data.random_split
### [torch.utils.data.random_split](https://pytorch.org/docs/1.13/data.html?highlight=torch+utils+data+random_split#torch.utils.data.random_split)

```python
torch.utils.data.random_split(dataset,
                            lengths,
                            generator=<torch._C.Generator object>)
```

### [paddle.io.random_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/random_split_cn.html)

```python
paddle.io.random_split(dataset,
                    lengths,
                    generator=None)
```

两者参数和用法完全一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dataset          | dataset            | 此参数必须是 paddle.io.Dataset 或 paddle.io.IterableDataset 的一个子类实例或实现了 __len__ 的 Python 对象，用于生成样本下标。默认值为 None。  |
| lengths         | lengths         | 总和为原数组长度的，子集合长度数组。 |
| generator         | generator         |   指定采样 data_source 的采样器。默认值为 None。 |
