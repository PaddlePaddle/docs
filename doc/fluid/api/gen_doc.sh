#!/bin/bash
python gen_doc.py layers --submodules control_flow device io nn ops tensor learning_rate_scheduler detection metric_op > layers.rst

for module in data_feeder dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler recordio_writer backward average profiler unique_name
do
  python gen_doc.py ${module} > ${module}.rst
done

python gen_doc.py "" > fluid.rst
