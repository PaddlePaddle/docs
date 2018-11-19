.. _api_guide_rnn:

####
rnn
####

循环神经网络是NLP、语音等相关领域大量应用网络结构，关于rnn的lstm的细节介绍请参考 `LSTM介绍 <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_ 

目前paddle支持两个方式的rnn实现：1）LOD方式，对于一个batch内部，长度差异非常大的数据，我们建议用户使用LOD的方式，LoD-tensor相关的说明见，`LoD tensor <http://paddlepaddle.org/documentation/docs/en/1.1/user_guides/howto/prepare_data/lod_tensor.html>`_ ; 2） Padding 方式，Padding方式是在短句子后面（或前面）加上padding id，使得一个batch 内的所有句子长度一致

1. LoD 方式：
---------------------

LoD方式支持的输入输入必须为LoD tensor。目前支持两种方式的LoD rnn使用方式

-1.1 直接使用现有的op，目前实现的op包含dynamic_lstm和dynamic_gru
  使用方法如下
  
  dynamic_gru :ref:`api_fluid_layers_dynamic_gru`

  dynamic_lstm :ref:`api_fluid_layers_dynamic_lstm`


-1.2 如果上面的op不能够满足用户的需求，用户可以定义直接的rnn的结构
  使用方法如下

  DynamicRNN :ref:`api_fluid_layers_DynamicRNN`

2. Padding 方式
---------------------

Padding方式适合一个batch内，句子长度差异大不，通过padding 方式使得句子长度变得一致

-1.1 对于定长的网络，可以使用for循环的方式实现，参考代码如下 `lm_model <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/language_model/lstm/lm_model.py>`_
  参考函数
  def encoder_static(input_embedding, len=10, init_hidden=None, init_cell=None):

-1.2 对于变长的网络，可以使用StaticRNN op来进行实现, 使用op为 :ref: `api_fluid_layers_StaticRNN` 参考代码如下 `lm_model <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/language_model/lstm/lm_model.py>`_
  参考函数
  def padding_rnn(input_embedding, len=10, init_hidden=None, init_cell=None):
