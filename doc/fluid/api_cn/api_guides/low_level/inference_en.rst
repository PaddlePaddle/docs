..  _api_guide_inference_en:

#################
Inference Engine
#################

Inference engine provides store inference model :ref:`api_fluid_io_save_inference_model` and load inference model :ref:`api_fluid_io_load_inference_model` .

Stored format of Inference Model
===================================

There are two stored formats of inference model, which are controlled by :code:`model_filename` of :ref:`api_fluid_io_save_inference_model` and :code:`params_filename` of :ref:`api_fluid_io_load_inference_model` respectively.

- Parameters are stored into different files, such as :code:`model_filename` set as :code:`None` and :code:`params_filename` set as :code:`None`

  .. code-block:: bash

      ls recognize_digits_conv.inference.model/*
      __model__ conv2d_1.w_0 conv2d_2.w_0 fc_1.w_0 conv2d_1.b_0 conv2d_2.b_0 fc_1.b_0

- Parameters are stored into the same file,such as :code:`model_filename` as :code:`None` and :code:`params_filename` as :code:`__params__`

  .. code-block:: bash

      ls recognize_digits_conv.inference.model/*
      __model__ __params__

Store Inference model
===============================

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'], 
        target_vars=[predict_var], executor=exe)

In this example, :code:`fluid.io.save_inference_model` will cut :code:`fluid.Program` into useful parts for inference :code:`predict_var` .
After the cut, :code:`program` will be preserved under :code:`./infer_model/__model__` while parameters will be preserved into independent files under :code:`./infer_model` .

Load Inference Model
=====================

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    [inference_program, feed_target_names, fetch_targets] = 
        fluid.io.load_inference_model(dirname=path, executor=exe)
    results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)

In this example, at first we call:code:`fluid.io.load_inference_model` to get inferenced :code:`program` , :code:`variable` name of input data and :code:`variable` of output;
then call :code:`executor` to run inferenced :code:`program` to get inferenced result.