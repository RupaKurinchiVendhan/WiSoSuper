import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == "__main__":
    sr_data = np.load('PhIREGAN\data_out\wind-20210721-124411\dataSR.npy')
    color_sr_data = sr_data[0,:,:,0] 
    plt.imsave("wind-20210721-124411_1.png", color_sr_data, origin='lower', format="png")
    # print(color_sr_data.shape)
    # color_sr_data = cv2.cvtColor(color_sr_data, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('solar-20210715-170358 (K=5).png', color_sr_data)