export MASTER=spark://$(hostname):7077
# export SPARK_WORKER_INSTANCES=5
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
# export TOTAL_CORES=3
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
# export TFoS_HOME=.
export TFoS_HOME=/Users/yongxiyang/Desktop/Tensorflow-2.0-try
# export SPARK_HOME=/home/sdev/yongxi/env/spark-2.4.4-bin-hadoop2.7
export SPARK_HOME=/Users/yongxiyang/Desktop/spark-2.4.4-bin-hadoop2.7

# absolute path and root

# confirm that data was generated
ls -lR ${TFoS_HOME}/data/mnist/csv

# remove any old artifacts
rm -rf ${TFoS_HOME}/mnist_model
rm -rf ${TFoS_HOME}/mnist_export
rm -rf ${TFoS_HOME}/logs

# train
${SPARK_HOME}/bin/spark-submit \
                    --master ${MASTER} \
                    --conf spark.cores.max=${TOTAL_CORES} \
                    --conf spark.task.cpus=${CORES_PER_WORKER} \
                    --num-executors ${SPARK_WORKER_INSTANCES} \
                    ${TFoS_HOME}/TBCallbacks.py \
                                    --cluster_size ${SPARK_WORKER_INSTANCES} \
                                    --epochs 5 \
                                    --images_labels ${TFoS_HOME}/data/mnist/csv/train \
                                    --model_dir ${TFoS_HOME}/mnist_model \
                                    --export_dir ${TFoS_HOME}/mnist_export \
                                    --tensorboard \
                                    --logdir ${TFoS_HOME}/logs
                    # ${TFoS_HOME}/examples/mnist/keras/mnist_spark.py \
                    # --executor-cores 1 \
                    # ${TFoS_HOME}/mnist_spark.py \
                    # ${TFoS_HOME}/try_spark.py \

