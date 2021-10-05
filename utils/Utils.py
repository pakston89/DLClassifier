import tensorflow as tf

#Getting number of GPUs working
class Utils():

    def gpuChecker():
        print("GPUs:", len(tf.config.experimental.list_physical_devices('GPU')))