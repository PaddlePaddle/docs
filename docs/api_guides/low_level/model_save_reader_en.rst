..  _api_guide_model_save_reader_en:

#######################
Save and Load a Model
#######################

To save and load model, there are eight APIs playing an important role:
:code:`fluid.io.save_vars`, :code:`fluid.io.save_params`, :code:`fluid.io.save_persistables`, :code:`fluid.io.save_inference_model`, :code:`fluid.io.load_vars`, :code:`fluid.io.load_params`, :code:`fluid.io.load_persistables` and :code:`fluid.io.load_inference_model` .

Variables, Persistables and Parameters
================================================

In :code:`Paddle` , every input and output of operator( :code:`Operator` ) is a variable( :code:`Variable` ), and parameter( :code:`Parameter` ) is a derived class of Variable( :code:`Variable` ). Persistables (:code:`Persistables`) are variables that won't be deleted after each iteration. Parameter is a kind of persistable variable which will be updated by optimizer ( :ref:`api_guide_optimizer_en` ) after each iteration. Training of neural network in essence is to update parameters.

Introduction to APIs for saving a model
========================================

- :code:`fluid.io.save_vars`: Variables are saved in specified directory by executor( :ref:`api_guide_executor_en` ). There are two ways to save variables:

  1）Set :code:`vars` in the API to assign the variable list to be saved.

  2）Assign an existed program( :code:`Program` ) to :code:`main_program` in the API, and then all variables in the program will be saved.

  The first one has a higher priority than the second one.

  For API Reference , please refer to :ref:`api_fluid_io_save_vars`.

- :code:`fluid.io.save_params`: Set :code:`main_program` in the API with the model Program( :code:`Program` ). This API will filter all parameters( :code:`Parameter` ) of targeted program and save them in folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  For API Reference , please refer to :ref:`api_fluid_io_save_params`.

- :code:`fluid.io.save_persistables`: :code:`main_program` of API assigns program( :code:`Program` ). This API will filter all persistables( :code:`persistable==True` ) of targeted program and save them in folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  For API Reference, please refer to :ref:`api_fluid_io_save_persistables`.

- :code:`fluid.io.save_inference_model`: please refer to  :ref:`api_guide_inference_en`.

Introduction to APIs for loading a model
========================================

- :code:`fluid.io.load_vars`: Executor( :code:`Executor` ) loads variables into the target directory. There are two ways to load variables:

  1）:code:`vars` in the API assigns variable list to be loaded.

  2）Assign an existed program( :code:`Program` ) to the :code:`main_program` field in the API, and then all variables in the program will be loaded.

  The first loading method has higher priority than the second one.

  For API Reference, please refer to :ref:`api_fluid_io_load_vars`.

- :code:`fluid.io.load_params`: This API filters all parameters( :code:`Parameter` ) in program assigned by :code:`main_program` and load these parameters from folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  For API Reference, please refer to :ref:`api_fluid_io_load_params` .

- :code:`fluid.io.load_persistables`:This API filters all persistables( :code:`persistable==True` ) in program assigned by :code:`main_program` and load these persistables from folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  For API Reference, please refer to :ref:`api_fluid_io_load_persistables` .

-  :code:`fluid.io.load_inference_model`: please refer to :ref:`api_guide_inference_en` .
