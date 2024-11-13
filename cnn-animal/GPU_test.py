import tensorflow as tf

# 檢查是否能夠使用 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 檢查 TensorFlow 版本
print("TensorFlow version: ", tf.__version__)

# 測試簡單的運算
hello = tf.constant('Hello, TensorFlow!')
print(hello)
