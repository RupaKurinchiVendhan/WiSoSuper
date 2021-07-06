import utils

if __name__ == '__main__':
    data = utils.csv_to_np('S:\Research\Spatio-temporal-SR-of-Wind-and-Solar-Data\get_data\WIND_DATA.csv')
    utils.generate_TFRecords('wind.tfrecord', data, mode='train', K=0.5)