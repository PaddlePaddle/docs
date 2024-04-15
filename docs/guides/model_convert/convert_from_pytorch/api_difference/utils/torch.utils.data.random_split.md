## [ 参数完全一致 ]torch.utils.data.random_split
### [torch.utils.data.random_split](https://pytorch.org/docs/stable/data.html?highlight=torch+utils+data+random_split#torch.utils.data.random_split)

```python
torch.utils.data.random_split(dataset,
                            lengths,
                            generator=<torch._C.Generator object>)
```

### [paddle.io.random_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/random_split_cn.html)

```python
paddle.io.random_split(dataset,
                    lengths,
                    generator=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                  |
| ------------- | ------------ |---------------------------------------------------------------------|
| dataset          | dataset            | 表示可迭代数据集。                                                           |
| lengths         | lengths         | 可为子集合长度列表，列表总和为原数组长度。也可为子集合所占比例列表，列表总和为 1.0。   |
| generator         | generator         | 指定采样 data_source 的采样器。默认值为 None。                                    |
