set -ex
# set environment variables (if not already done)
# export PYTHON_ROOT=./Python
export LD_LIBRARY_PATH=${PATH}
# export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
# export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
# export PATH=${PYTHON_ROOT}/bin/:$PATH
export QUEUE=adx
export SPARK_HOME=/home/sdev/yongxi/spark-2.4.4-bin-hadoop2.7

# set paths to libjvm.so, libhdfs.so, and libcuda*.so
#export LIB_HDFS=/opt/cloudera/parcels/CDH/lib64                      # for CDH (per @wangyum)
export LIB_HDFS=/usr/lib/ams-hbase/lib/hadoop-native                  # path to libhdfs.so, for TF acccess to HDFS
export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server                        # path to libjvm.so
# export LIB_CUDA=/usr/local/cuda-7.5/lib64                             # for GPUs only

# on the cluster the path for lihdfs.so and libjvm.so
# /usr/hdp/2.5.6.0-40/usr/lib/libhdfs.so
# /usr/lib/ams-hbase/lib/hadoop-native/libhdfs.so
export HADOOP_HDFS_HOME=/usr/hdp/2.5.6.0-40/hadoop-hdfs/*

TFCONNECTOR=hdfs:///user-profile/yongxi/spark/jars/spark-tensorflow-connector_2.11-1.15.0.jar
TFHADOOP=hdfs:///user-profile/yongxi/spark/jars/tensorflow-hadoop-1.15.0.jar
CONDAENV=tf2dis

INPUT_DATA=hdfs:///user-profile/yongxi/spark/input/mnist/csv/train
MODEL_DIR=hdfs:///user-profile/yongxi/spark/tfoutput/mnist_model
# for CPU mode:
# export QUEUE=default
# remove references to $LIB_CUDA

# # save images and labels as CSV files
# ${SPARK_HOME}/bin/spark-submit \
# --master yarn \
# --deploy-mode cluster \
# --queue ${QUEUE} \
# --num-executors 4 \
# --executor-memory 4G \
# --archives hdfs:///user/${USER}/Python.zip#Python,mnist/mnist.zip#mnist \
# --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_CUDA \
# --driver-library-path=$LIB_CUDA \
# TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
# --output mnist/csv \
# --format csv

# # save images and labels as TFRecords (OPTIONAL)
# ${SPARK_HOME}/bin/spark-submit \
# --master yarn \
# --deploy-mode cluster \
# --queue ${QUEUE} \
# --num-executors 4 \
# --executor-memory 4G \
# --archives hdfs:///user/${USER}/Python.zip#Python,mnist/mnist.zip#mnist \
# --jars hdfs:///user/${USER}/tensorflow-hadoop-1.0-SNAPSHOT.jar \
# --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_CUDA \
# --driver-library-path=$LIB_CUDA \
# TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
# --output mnist/tfr \
# --format tfr

# For TensorFlow 2.x (git checkout master)
hadoop fs -rm -r ${MODEL_DIR}
${SPARK_HOME}/bin/spark-submit \
                    --master yarn \
                    --deploy-mode cluster \
                    --queue ${QUEUE} \
                    --num-executors 5 \
                    --executor-memory 2G \
                    --conf spark.dynamicAllocation.enabled=false \
                    --conf spark.yarn.maxAppAttempts=1 \
                    --archives "../${CONDAENV}.zip#${CONDAENV}_zip" \
                    --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
                    --jars ${TFCONNECTOR}
                    --jars ${TFHADOOP}
                    ./examples/mnist/keras/mnist_spark.py \
                    --images_labels ${INPUT_DATA} \
                    --model_dir ${MODEL_DIR}
