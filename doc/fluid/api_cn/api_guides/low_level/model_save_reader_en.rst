..  _api_guide_model_save_reader_en:

#######################
Save and Load of Model
########################

The save and load of model are mainly involved with eight APIs below:
:code:`fluid.io.save_vars`, :code:`fluid.io.save_params`, :code:`fluid.io.save_persistables`, :code:`fluid.io.save_inference_model`, :code:`fluid.io.load_vars`, :code:`fluid.io.load_params`, :code:`fluid.io.load_persistables` and :code:`fluid.io.load_inference_model` .

Variable,Persistables and Parameters
========================================

In :code:`Paddle` , every input and output of operator( :code:`Operator` ) is a variable( :code:`Variable` ),but parameter( :code:`Parameter` ) is child class of variable( :code:`Variable` ). Persistable (:code:`Persistables`) is a varaible that won't be deleted after each interation. Parameter is a kind of persistable variable which will be updated by optimizer ( :ref:`api_guide_optimizer` ) after each iteration. To train neural network in fact is to update parameters.

Introduction of Save API of model
==================================

- :code:`fluid.io.save_vars`:variables are saved in targeted directory with executor( :ref:`api_guide_executor` ). There are two ways to save variable:

  1）:code:`vars` of API assigns variable list to be saved.

  2）Assign an existed program( :code:`Program` ) to :code:`main_program` of API, then with all variables in the program saved.

  The former is prior to the latter.

  About API Reference , please refer to :ref:`api_fluid_io_save_vars`.

- :code:`fluid.io.save_params`: :code:`main_program` of API assigns program( :code:`Program` ). This API will filter all parameters( :code:`Parameter` ) of targeted program and save them in folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  About API Reference , please refer to :ref:`api_fluid_io_save_params`.

- :code:`fluid.io.save_persistables`: :code:`main_program` of API assigns program( :code:`Program` ). This API will filter all persistables( :code:`persistable==True` ) of targeted program and save them in folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  About API Reference, please refer to :ref:`api_fluid_io_save_persistables`.

- :code:`fluid.io.save_inference_model`: please refer to  :ref:`api_guide_inference`.  

Introduction of Load API of model
====================================

- :code:`fluid.io.load_vars`: Executor( :code:`Executor` ) loads variables in targeted directory. There are two ways to load variables:
  
  1）:code:`vars` of API assigns variable list to be loaded.
  
  2）Assign an existed program( :code:`Program` ) to :code:`main_program` of API, then with all variables in the program loaded.

  The former is prior to the latter.

  About API Reference, please refer to :ref:`api_fluid_io_load_vars`.

- :code:`fluid.io.load_params`: This API filters all parameters( :code:`Parameter` ) in program assigned by :code:`main_program` and load these parameters from folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  About API Reference, please refer to :ref:`api_fluid_io_load_params` .

- :code:`fluid.io.load_persistables`:This API filter all persistables( :code:`persistable==True` ) in program assigned by :code:`main_program` and load these persistables from folder assigned by :code:`dirname` or file assigned by :code:`filename` .

  About API Reference, please refer to :ref:`api_fluid_io_load_persistables` .

-  :code:`fluid.io.load_inference_model`: please refer to :ref:`api_guide_inference` .