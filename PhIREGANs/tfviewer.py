import tensorflow as tf

if __name__ == '__main__':
    for example in tf.python_io.tf_record_iterator("example_data\wind_LR-MR.tfrecord"):
        print(tf.train.Example.FromString(example))