set -ex

# set environment variables (if not already done)
# conda env
CONDAENV=tf2dis

# /usr/hdp/2.5.6.0-40/hadoop/lib/native
# /usr/hdp/2.5.6.0-40/hadoop/libexec/hadoop-config.sh

# export HADOOP_HDFS_HOME=/usr/hdp/2.5.6.0-40/hadoop-hdfs
export HADOOP_HOME=/usr/hdp/2.5.6.0-40/hadoop
# export CLASSPATH=$(hadoop classpath --glob)
# export LD_LIBRARY_PATH=${PATH}
export PYSPARK_PYTHON="./${CONDAENV}_zip/${CONDAENV}/bin/python"
export QUEUE=adx
export SPARK_HOME=/home/sdev/yongxi/spark-2.4.4-bin-hadoop2.7

# set paths to libjvm.so, libhdfs.so, and libcuda*.so
export LIB_HDFS=/home/sdev/yongxi/env/tfhdfs/lib                                     # path to libhdfs.so, for TF acccess to HDFS
# already upto hdfs:///user-profile/yongxi/spark/env/tfhdfs/lib
# export LIB_JVM=hdfs:///user-profile/yongxi/spark/env/tfjvm                           # path to libjvm.so
# # set paths to libjvm.so, libhdfs.so, and libcuda*.so
# export LIB_HDFS=$HADOOP_PREFIX/lib/native/Linux-amd64-64
# # already upto hdfs:///user-profile/yongxi/spark/env/tfjvm
export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server

# on the cluster the path for lihdfs.so and libjvm.so
# /usr/hdp/2.5.6.0-40/usr/lib/libhdfs.so
# /usr/lib/ams-hbase/lib/hadoop-native/libhdfs.so

export HADOOP_USER_NAME=profile
# export CLASSPATH=$(hadoop classpath --glob)

# jar Package on the air
TFCONNECTOR=hdfs:///user-profile/yongxi/spark/env/jars/spark-tensorflow-connector_2.11-1.15.0.jar
TFHADOOP=hdfs:///user-profile/yongxi/spark/env/jars/tensorflow-hadoop-1.15.0.jar


# spark configuration
SPARK_WORKER_INSTANCES=5
EXECUTOR_MEMORY=2G

# Train configuration
EPOCHS=2

# Input and output and not "hdfs://" pre-ffix
# and must obtain the write permission all the way of the path.
# because tensorflow will be create recursively files and paths.
INPUT_DATA=/user-profile/yongxi/spark/input/mnist/csv/train
MODEL_DIR=hdfs://opera/tmp/yongxi/tfoutput/mnist_model
EXPORT_DIR=hdfs://opera/tmp/yongxi/tfoutput/mnist_export

sudo -u ${HADOOP_USER_NAME} hadoop fs -rm -r -f -skipTrash ${MODEL_DIR}/*
sudo -u ${HADOOP_USER_NAME} hadoop fs -rm -r -f -skipTrash ${EXPORT_DIR}/*

sudo -u ${HADOOP_USER_NAME} ${SPARK_HOME}/bin/spark-submit \
                    --master yarn \
                    --deploy-mode cluster \
                    --queue ${QUEUE} \
                    --num-executors ${SPARK_WORKER_INSTANCES} \
                    --executor-memory ${EXECUTOR_MEMORY} \
                    --conf spark.dynamicAllocation.enabled=false \
                    --conf spark.yarn.maxAppAttempts=1 \
                    --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
                    --conf spark.executorEnv.HADOOP_USER_NAME=${HADOOP_USER_NAME} \
                    --conf spark.network.timeout=3600s \
                    --conf spark.executorEnv.CLASSPATH=${CLASSPATH} \
                    --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYSPARK_PYTHON}" \
                    --archives "../${CONDAENV}.zip#${CONDAENV}_zip" \
                    --jars ${TFCONNECTOR},${TFHADOOP} \
                    mnist_spark.py \
                        --cluster_size ${SPARK_WORKER_INSTANCES} \
                        --epochs ${EPOCHS} \
                        --images_labels ${INPUT_DATA} \
                        --model_dir ${MODEL_DIR} \
                        --export_dir ${EXPORT_DIR}
                    # try_spark.py \
                    # --conf spark.task.maxFailures=1 \
