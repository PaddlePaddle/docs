## [ torch 参数更多 ]torch.utils.data.DistributedSampler
### [torch.utils.data.DistributedSampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler)

```python
torch.utils.data.DistributedSampler(dataset,
                                    num_replicas=None,
                                    rank=None,
                                    shuffle=True,
                                    seed=0,
                                    drop_last=False)
```

### [paddle.io.DistributedBatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/DistributedBatchSampler_cn.html#distributedbatchsampler)

```python
paddle.io.DistributedBatchSampler(dataset=None,
                                    batch_size,
                                    num_replicas=None,
                                    rank=None,
                                    shuffle=False,
                                    drop_last=False)
```

PyTorch 参数更多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ----- | ---------- | ---------- |
| dataset             | dataset   | 被采样的数据集。                   |
| -             | batch_size   | 每 mini-batch 中包含的样本数，PyTorch 无此参数，Paddle 需设置为 1。                   |
| num_replicas             | num_replicas   | 分布式训练时的进程个数。                   |
| rank             | rank   | num_replicas 个进程中的进程序号。                   |
| shuffle             | shuffle   | 是否需要在生成样本下标时打乱顺序。与 PyTorch 默认值不同， Paddle 应设置为 `True`。                    |
| seed          | -            | 如果 shuffle=True，则使用随机种子对采样器进行随机排序,此数字在分布式组中的所有进程中应相同，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| drop_last             | drop_last   | 是否需要丢弃最后无法凑整一个 mini-batch 的样本。                   |
