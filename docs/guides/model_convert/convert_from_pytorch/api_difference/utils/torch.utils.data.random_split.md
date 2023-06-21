## [ 参数不一致 ]torch.utils.data.random_split
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

两者参数除 lengths 外用法一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                                  |
| ------------- | ------------ |---------------------------------------------------------------------|
| dataset          | dataset            | 表示可迭代数据集。                                                           |
| lengths         | lengths         | PyTorch:总和为原数组长度或 1.0，子集合长度或总长度比例数组。PaddlePaddle: 总和为原数组长度的，子集合长度数组。 |
| generator         | generator         | 指定采样 data_source 的采样器。默认值为 None。                                    |

### 转写示例
当参数 lenghts 为总长度的比例数组时，转写如下:
```python
# Pytorch 写法
lengths = [0.3, 0.3, 0.4]
datasets = torch.utils.data.random_split(range(30),
                                        lengths,
                                        generator=<torch._C.Generator object>)

# Paddle 写法
lengths = [0.3, 0.3, 0.4]
lengths = [length * datasets.__len__() for length in lengths]
datasets = paddle.io.random_split(dataset,
                                  lengths,
                                  generator=None)
```
