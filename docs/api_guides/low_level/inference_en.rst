..  _api_guide_inference_en:

#################
Inference Engine
#################

Inference engine provides interfaces to save inference model :ref:`api_fluid_io_save_inference_model` and load inference model :ref:`api_fluid_io_load_inference_model` .

Format of Saved Inference Model
=====================================

There are two formats of saved inference model, which are controlled by :code:`model_filename`  and :code:`params_filename`  parameters in the two interfaces above.

- Parameters are saved into independent separate files, such as :code:`model_filename` set as :code:`None` and :code:`params_filename` set as :code:`None`

  .. code-block:: bash

      ls recognize_digits_conv.inference.model/*
      __model__ conv2d_1.w_0 conv2d_2.w_0 fc_1.w_0 conv2d_1.b_0 conv2d_2.b_0 fc_1.b_0

- Parameters are saved into the same file, such as :code:`model_filename` set as :code:`None` and :code:`params_filename` set as :code:`__params__`

  .. code-block:: bash

      ls recognize_digits_conv.inference.model/*
      __model__ __params__

Save Inference model
===============================

To save an inference model, we normally use :code:`fluid.io.save_inference_model` to tailor the default :code:`fluid.Program` and only keep the parts useful for predicting :code:`predict_var`.
After being tailored, :code:`program` will be saved under :code:`./infer_model/__model__` while the parameters will be saved into independent files under :code:`./infer_model` .

Sample Code:

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
        target_vars=[predict_var], executor=exe)


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

In this example, at first we call :code:`fluid.io.load_inference_model` to get inference :code:`inference_program` , :code:`feed_target_names`-name of input data and :code:`fetch_targets` of output;
then call :code:`executor` to run inference :code:`inference_program` to get inferred result.
