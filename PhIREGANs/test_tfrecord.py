import tensorflow as tf
tf.enable_eager_execution()
from tfrecord_lite import decode_example

tf.logging.set_verbosity(tf.logging.ERROR)

def _parse_train_(serialized_example, mu_sig=None):
    '''
        Parser data from TFRecords for the models to read in for (pre)training

        inputs:
            serialized_example - batch of data drawn from tfrecord
            mu_sig             - mean, standard deviation if known

        outputs:
            idx     - array of indicies for each sample
            data_LR - array of LR images in the batch
            data_HR - array of HR images in the batch
    '''

    feature = {'index': tf.FixedLenFeature([], tf.int64),
                'data_LR': tf.FixedLenFeature([], tf.string),
                'h_LR': tf.FixedLenFeature([], tf.int64),
                'w_LR': tf.FixedLenFeature([], tf.int64),
                # 'data_HR': tf.FixedLenFeature([], tf.string),
                # 'h_HR': tf.FixedLenFeature([], tf.int64),
                # 'w_HR': tf.FixedLenFeature([], tf.int64),
                'c': tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(serialized_example, feature)

    idx = example['index']

    h_LR, w_LR = example['h_LR'], example['w_LR']
    # h_HR, w_HR = example['h_HR'], example['w_HR']
    c = example['c']

    print(f'index: {idx}')
    print(f'h_LR: {h_LR}, w_LR: {w_LR}')
    # print(f'h_HR: {h_HR}, w_HR: {w_HR}')
    print(f'c: {c}')

    data_LR = tf.decode_raw(example['data_LR'], tf.float64)
    # data_HR = tf.decode_raw(example['data_HR'], tf.float64)

    print(f'Data LR decoded shape: {data_LR.shape}')
    # print(f'Data HR decoded shape: {data_HR.shape}')

    print(f'LR Data: {data_LR[0:10]}')
    # print(f'HR Data: {data_HR[0:10]}')

    # print(f'Data LR Shape: {data_LR.shape}')

    data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
    # data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

    # print(f'Data LR Shape, Data HR Shape: {data_LR.shape}, {data_HR.shape}')

    if mu_sig is not None:
        data_LR = (data_LR - mu_sig[0])/mu_sig[1]
        # data_HR = (data_HR - mu_sig[0])/mu_sig[1]

    return idx, data_LR

for _ in range(10):
    ds1 = tf.python_io.tf_record_iterator('example_data/wind_LR-MR.tfrecord')
    # ds2 = tf.python_io.tf_record_iterator('example_data/wind9.tfrecord')

    # ds1 = tf.python_io.tf_record_iterator('example_data/wind12.tfrecord')
    ds2 = tf.python_io.tf_record_iterator('example_data/us_wind_HR.tfrecord')

    print('-------- Parsing DS1 ------------')
    _parse_train_(next(ds1))

    print('-------- Parsing DS2 ------------')
    _parse_train_(next(ds2))