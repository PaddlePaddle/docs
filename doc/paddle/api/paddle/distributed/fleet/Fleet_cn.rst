.. _cn_api_distributed_fleet_Fleet:

Fleet
-------------------------------


.. py:class:: paddle.distributed.fleet.Fleet




.. py:method:: init(role_maker=None, is_collective=False)


.. py:method:: is_first_worker()


.. py:method:: worker_index()


.. py:method:: worker_num()


.. py:method:: is_worker()


.. py:method:: worker_endpoints(to_string=False)


.. py:method:: server_num()


.. py:method:: server_index()


.. py:method:: server_endpoints(to_string=False)


.. py:method:: is_server()


.. py:method:: barrier_worker()


.. py:method:: init_worker()


.. py:method:: init_server(*args, **kwargs)


.. py:method:: run_server()


.. py:method:: stop_worker()


.. py:method:: save_inference_model(executor, dirname, feeded_var_names, target_vars, main_program=None, export_for_deployment=True)


.. py:method:: save_persistables(executor, dirname, main_program=None)


.. py:method:: distributed_optimizer(optimizer, strategy=None)


.. py:method:: distributed_model(model)


.. py:method:: state_dict()


.. py:method:: set_lr(value)


.. py:method:: get_lr()


.. py:method:: step()


.. py:method:: clear_grad()


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)


.. py:attribute:: util


