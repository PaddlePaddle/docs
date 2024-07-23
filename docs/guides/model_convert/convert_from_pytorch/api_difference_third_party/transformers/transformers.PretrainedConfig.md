## [torch 参数更多]transformers.PretrainedConfig

### [transformers.PretrainedConfig](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/configuration#transformers.PretrainedConfig)

```python
transformers.PretrainedConfig(*kwargs)
```

### [paddlenlp.transformers.PretrainedConfig](https://github.com/PaddlePaddle/PaddleNLP/blob/57000fa12ce67024238f0b56a6fde63c592c54ce/paddlenlp/transformers/configuration_utils.py#L317)

```python
paddlenlp.transformers.PretrainedConfig(*kwargs)
```

两者功能一致，但 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| transformers                         | PaddlePaddle                   | 备注      |
| -------------------------------------| ------------------------------ | -------- |
| name_or_path                         | name_or_path                   | 传递给 from_pretrained 的模型名称或模型所在路径。 |
| output_attentions                    | output_attentions              | 是否返回注意力层的注意力张量。 |
| output_hidden_states                 | output_hidden_states           | 是否返回注意力层的隐藏层张量。 |
| return_dict                          | return_dict                    | 是否返回 dict 而不是 tuple。 |
| is_encoder_decoder                   | is_encoder_decoder             | 模型是否为 encoder-decoder 架构。 |
| is_decoder                           | min_length                     | 模型是否为 decoder only 架构。 |
| cross_attention_hidden_size          | cross_attention_hidden_size    | cross_attention 层隐藏层维数。 |
| add_cross_attention                  | add_cross_attention            | 是否增加 cross_attention 层。 |
| tie_encoder_decoder                  | tie_encoder_decoder            | encoder 与 decoder 的权重是否关联。|
| prune_heads                          | prune_heads                    | 修剪模型的 heads。 |
| chunk_size_feed_forward              | chunk_size_feed_forward        | 前馈层的 chunk size 。 |
| max_length                           | max_length                     | 最大生成长度。 |
| min_length                           | min_length                     | 最小生成长度。 |
| early_stopping                       | early_stopping                 | 早停是否开启。 |
| do_sample                            | do_sample                      | 是否进行采样。 |
| num_beams                            | num_beams                      | beams for beam search。 |
| num_beam_groups                      | num_beam_groups                | beams 划分的组数。 |
| diversity_penalty                    | diversity_penalty              | 分散惩罚系数。 |
| temperature                          | temperature                    | 用于控制下个 token 生成的参数。 |
| top_k                                | top_k                          | top_k 算法的 k 值。 |
| top_p                                | top_p                          | top_p 算法的 p 值。 |
| typical_p                            | -                              | 局部典型度量的参数，Paddle 无此参数，暂无转写方式。 |
| repetition_penalty                   | repetition_penalty             | 重复惩罚参数。 |
| length_penalty                       | length_penalty                 | 长度重复惩罚参数。 |
| no_repeat_ngram_size                 | no_repeat_ngram_size           | ngram 在给定长度内不可重复。 |
| encoder_no_repeat_ngram_size         | encoder_no_repeat_ngram_size   | encoder 中指定 size 内不能出现同一个 ngram。 |
| bad_words_ids                        | bad_words_ids                  | 不允许生成的 id 列表。 |
| num_return_sequences                 | num_return_sequences           | 为 batch 中每个序列独立计算返回序列的个数。|
| output_scores                        | output_scores                  | 是否返回注意力层的得分张量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| return_dict_in_generate              | return_dict_in_generate        | 是否返回 dict 而不是 tuple，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| forced_bos_token_id                  | forced_bos_token_id            | 强制生成的 bos token 的 id。 |
| forced_eos_token_id                  | forced_eos_token_id            | 强制生成的 eos token 的 id。 |
| remove_invalid_values                | remove_invalid_values          | 是否移除无效值。 |
| architectures                        | architectures                  | 可共用预训练参数的模型架构。 |
| finetuning_task                      | finetuning_task                | 微调模型的任务名称。  |
| id2label                             | id2label                       | 索引到标签的映射。 |
| label2id                             | label2id                       | 标签到索引的映射。 |
| num_labels                           | num_labels                     | 模型最后一层使用的标签数。 |
| task_specific_params                 | task_specific_params           | 为当前任务额外指定的参数。 |
| problem_type                         | problem_type                   | 用于解决的问题类型。 |
| tokenizer_class                      | tokenizer_class                | tokenizer 的类别。 |
| prefix                               | prefix                         | 调用前增加的 prompt。 |
| pad_token_id                         | pad_token_id                   | padding token 的 id。 |
| bos_token_id                         | bos_token_id                   | beginning-of-sequence token 的 id。 |
| eos_token_id                         | eos_token_id                   | end-of-sequence  token 的 id。 |
| decoder_start_token_id               | decoder_start_token_id         | decoder 生成的第一个 token 的 id。 |
| torchscript                          | -                              | 模型是否和 torchscript 一起使用，Paddle 无此参数，暂无转写方式。|
| tie_word_embeddings                  | tie_word_embeddings            | input 和 output 的 word embedding 层参数是否绑定。 |
| torch_dtype                          | dtype                          | 模型参数的数据类型。 |
