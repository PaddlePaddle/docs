## [ 仅 API 调用方式不一致 ]torch.utils.data.SequentialSampler

### [torch.utils.data.SequentialSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler)

```python
torch.utils.data.SequentialSampler(data_source)
```

### [paddle.io.SequenceSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/SequenceSampler_cn.html#paddle.io.SequenceSampler)

```python
paddle.io.SequenceSampler(data_source)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
s = torch.utils.data.SequentialSampler(MyDataset())

# Paddle 写法
s = paddle.io.SequenceSampler(MyDataset())
```
