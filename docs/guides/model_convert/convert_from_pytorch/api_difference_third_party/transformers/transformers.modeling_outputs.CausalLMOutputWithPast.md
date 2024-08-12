## [参数完全一致]transformers.modeling_outputs.CausalLMOutputWithPast

### [transformers.modeling_outputs.CausalLMOutputWithPast](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast)

```python
transformers.modeling_outputs.CausalLMOutputWithPast(loss: Optional = None, logits: FloatTensor = None, past_key_values: Optional = None, hidden_states: Optional = None, attentions: Optional = None)
```

### [paddlenlp.transformers.model_outputs.CausalLMOutputWithPast](https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/model_outputs.py#L874)

```python
paddlenlp.transformers.model_outputs.CausalLMOutputWithPast(loss: Optional[paddle.Tensor] = None, logits: paddle.Tensor = None, past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None, hidden_states: Optional[Tuple[paddle.Tensor]] = None, attentions: Optional[Tuple[paddle.Tensor]] = None)
```

功能一致，参数完全一致，具体如下：

### 参数映射
| transformers      | PaddlePaddle                                          | 备注                                                    |
|-------------------|-------------------------------------------------------|---------------------------------------------------------|
| loss              | loss                                                  | 损失值。默认值为 None。                                 |
| logits            | logits                                                | 模型输出的 logits。默认值为 None。                       |
| past_key_values   | past_key_values                                       | 过去的键值对。默认值为 None。                           |
| hidden_states     | hidden_states                                         | 隐藏状态。默认值为 None。                               |
| attentions        | attentions                                            | 注意力权重。默认值为 None。                             |
