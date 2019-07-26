.. _cn_api_fluid_dygraph_save_persistables:

save_persistables
-------------------------------

.. py:function:: paddle.fluid.dygraph.save_persistables(model_dict, dirname='save_dir', optimizers=None)

该函数把传入的层中所有参数以及优化器进行保存。

``dirname`` 用于指定保存长期变量的目录。

参数:
 - **model_dict**  (dict of Parameters) – 参数将会被保存，如果设置为None，不会处理。
 - **dirname**  (str) – 目录路径
 - **optimizers**  (fluid.Optimizer|list(fluid.Optimizer)|None) –  要保存的优化器。 

返回: None
  
**代码示例**

.. code-block:: python
    
          import paddle.fluid as fluid

          ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)
          sgd = fluid.optimizer.SGD(learning_rate=0.01)
          x_data = np.arange(12).reshape(4, 3).astype('int64')
          y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
          x_data = x_data.reshape((-1, num_steps, 1))
          y_data = y_data.reshape((-1, 1))
          init_hidden_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
          init_cell_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
          x = to_variable(x_data)
          y = to_variable(y_data)
          init_hidden = to_variable(init_hidden_data)
          init_cell = to_variable(init_cell_data)
          dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                        init_cell)
          dy_loss.backward()
          sgd.minimize(dy_loss)
          ptb_model.clear_gradient()
          param_path = "./my_paddle_model"
          fluid.dygraph.save_persistables(ptb_model.state_dict(), dirname=param_path, sgd)
    
    





