export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
# export TOTAL_CORES=4
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export TFoS_HOME=/Users/yongxiyang/Desktop/Tensorflow-2.0-try
export SPARK_HOME=/Users/yongxiyang/Desktop/spark-2.4.4-bin-hadoop2.7

# absolute path and root

# confirm that data was generated
ls -lR ${TFoS_HOME}/data/mnist/csv

# remove any old artifacts
rm -rf ${TFoS_HOME}/mnist_model
rm -rf ${TFoS_HOME}/mnist_export

# train
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${TFoS_HOME}/try_spark.py \
                --epochs 2 \
                --cluster_size ${SPARK_WORKER_INSTANCES} \
                --images_labels ${TFoS_HOME}/data/mnist/csv/train \
                --model_dir ${TFoS_HOME}/mnist_model \
                --export_dir ${TFoS_HOME}/mnist_export

