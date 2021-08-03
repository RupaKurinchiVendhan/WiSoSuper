import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, log10
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error as mae
from PIL import Image


def read_image(path):
    '''Read image and return the image propertis.
    Parameters:
    path (string): Image path

    Returns:
    numpy.ndarray: Image exists in "path"
    list: Image size
    tuple: Image dimension (number of rows and columns)
    '''
    img = cv2.imread(path)
    size = img.shape
    dimension = (size[0], size[1])

    return img, size, dimension

def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    '''Resize image to a specificall scale of original image.
    Parameters:
    img (numpy.ndarray): Original image
    dimension (tuple): Original image dimension
    scale (int): Multiply the size of the original image

    Returns:
    numpy.ndarray: Resized image
    '''
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img


def W(x):
    '''Weight function that return weight for each distance point
    Parameters:
    x (float): Distance from destination point

    Returns:
    float: Weight
    '''
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a+2)*(pos_x**3)) - ((a+3)*(pos_x**2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x**3)) - (5*a*(pos_x**2)) + (8 * a * pos_x) - 4*a)
    else:
        return 0
        

def bilinear_interpolation(image, dimension):
    '''Bilinear interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension

    Returns:
    numpy.ndarray: Resized image
    '''
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    return new_image


def u(s, a): 
    if (abs(s) >= 0) & (abs(s) <= 1): 
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2): 
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a 
    return 0
  
  
# Padding 
def padding(img, H, W, C): 
    zimg = np.zeros((H+4, W+4, C)) 
    zimg[2:H+2, 2:W+2, :C] = img 
      
    # Pad the first/last two col and row 
    zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C] 
    zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :] 
    zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :] 
    zimg[0:2, 2:W+2, :C] = img[0:1, :, :C] 
      
    # Pad the missing eight points 
    zimg[0:2, 0:2, :C] = img[0, 0, :C] 
    zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C] 
    zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C] 
    zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C] 
    return zimg 
  
  
# Bicubic operation 
def bicubic(img, ratio, a): 
    
    # Get image size 
    H, W, C = img.shape 
      
    # Here H = Height, W = weight, 
    # C = Number of channels if the  
    # image is coloured. 
    img = padding(img, H, W, C) 
      
    # Create new image 
    dH = floor(H*ratio) 
    dW = floor(W*ratio) 
  
    # Converting into matrix 
    dst = np.zeros((dH, dW, 3))   
    # np.zeroes generates a matrix  
    # consisting only of zeroes 
    # Here we initialize our answer  
    # (dst) as zero 
  
    h = 1/ratio 
  
    print('Start bicubic interpolation') 
    print('It will take a little while...') 
    inc = 0
      
    for c in range(C): 
        for j in range(dH): 
            for i in range(dW): 
                
                # Getting the coordinates of the 
                # nearby values 
                x, y = i * h + 2, j * h + 2
  
                x1 = 1 + x - floor(x) 
                x2 = x - floor(x) 
                x3 = floor(x) + 1 - x 
                x4 = floor(x) + 2 - x 
  
                y1 = 1 + y - floor(y) 
                y2 = y - floor(y) 
                y3 = floor(y) + 1 - y 
                y4 = floor(y) + 2 - y 
                  
                # Considering all nearby 16 values 
                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]]) 
                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c], 
                                    img[int(y-y2), int(x-x1), c], 
                                    img[int(y+y3), int(x-x1), c], 
                                    img[int(y+y4), int(x-x1), c]], 
                                   [img[int(y-y1), int(x-x2), c], 
                                    img[int(y-y2), int(x-x2), c], 
                                    img[int(y+y3), int(x-x2), c], 
                                    img[int(y+y4), int(x-x2), c]], 
                                   [img[int(y-y1), int(x+x3), c], 
                                    img[int(y-y2), int(x+x3), c], 
                                    img[int(y+y3), int(x+x3), c], 
                                    img[int(y+y4), int(x+x3), c]], 
                                   [img[int(y-y1), int(x+x4), c], 
                                    img[int(y-y2), int(x+x4), c], 
                                    img[int(y+y3), int(x+x4), c], 
                                    img[int(y+y4), int(x+x4), c]]]) 
                mat_r = np.matrix( 
                    [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]]) 
                  
                # Here the dot function is used to get  
                # the dot product of 2 matrices 
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r) 

    return dst 


def bicubic_interpolation(img, dimension):
    '''Bicubic interpolation method to convert small size image to original size image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension

    Returns:
    numpy.ndarray: Resized image
    '''
    nrows = dimension[0]
    ncols = dimension[1]

    output = np.zeros((nrows, ncols, img.shape[2]), np.uint8)
    for c in range(img.shape[2]):
        for i in range(nrows):
            for j in range(ncols):
                xm = (i + 0.5) * (img.shape[0]/dimension[0]) - 0.5
                ym = (j + 0.5) * (img.shape[1]/dimension[1]) - 0.5

                xi = floor(xm)
                yi = floor(ym)

                u = xm - xi
                v = ym - yi

                out = 0
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if ((xi + n < 0) or (xi + n >= img.shape[1]) or (yi + m < 0) or (yi + m >= img.shape[0])):
                            continue

                        out += (img[xi+n, yi+m, c] * (W(u - n) * W(v - m)))

                output[i, j, c] = np.clip(out, 0, 255)

    return output


def psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    # MSE is zero means no noise is present in the signal and PSNR has no importance.
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mse(imageA, imageB):
 # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
 mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
 mse_error /= float(imageA.shape[0] * imageA.shape[1])
 # return the MSE. The lower the error, the more "similar" the two images are.
 return mse_error


def mae(imageA, imageB):
    mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float"))))
    mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
    if (mae < 0):
        return mae * -1
    else:
        return mae

def interpolate(filename):
    img, size, dimension = read_image(filename)

    # Change Image Size
    scale_percent = 20  # percent of original image size
    resized_img = image_change_scale(img, dimension, scale_percent)

    # Change image to original size using bilinear interpolation
    bil_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    bil_img_algo = bilinear_interpolation(resized_img, dimension)
    bil_img_algo = Image.fromarray(bil_img_algo.astype('uint8')).convert('RGB')

    # Change image to original size using bicubic interpolation
    # cubic_img_algo = bicubic(resized_img, 4, -1/4)
    cubic_img_algo = bicubic_interpolation(resized_img, dimension)
    cubic_img_algo = Image.fromarray(
        cubic_img_algo.astype('uint8')).convert('RGB')
    # cv2.imwrite('cubic.png', np.array(cubic_img_algo))
    # cv2.imwrite('bilinear.png', np.array(bil_img_algo))
    return np.array(bil_img_algo), np.array(cubic_img_algo)

def rotate(image_path, degrees_to_rotate, saved_location):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location)

def flip_image(image_path, saved_location):
    """
    Flip or mirror the image
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)

def main(filenameA, gan):
    imgA, sizeA, _ = read_image(filenameA)
    print(f"Image Size is: {sizeA}")
    # rotate(gan, 180, gan)
    # flip_image(gan, gan)
    imgB, _, _ = read_image(gan)
    imgC, imgD = interpolate(filenameA)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    grayC = cv2.cvtColor(np.array(imgC), cv2.COLOR_BGR2GRAY)
    grayD = cv2.cvtColor(np.array(imgD), cv2.COLOR_BGR2GRAY)

    print("------- GANs Metrics -------")
    # Calculate and print the PSNR value
    psnr_val = psnr(imgA, imgB)
    print(f"PSNR: {psnr_val}")
    # Calculate and print the SSIM value
    ssim_val = ssim(grayA, grayB)
    print(f"SSIM: {ssim_val}")
    # Calculate and print the MSE value
    mse_val = mse(grayA, grayB)
    print(f"MSE: {mse_val}")
    # Calculate and print the MAE value
    mae_val = mae(imgA, imgB)
    print(f"MAE: {mae_val}")

    print("------- Bilinear Metrics -------")
    # Calculate and print the PSNR value
    psnr_val = psnr(imgA, imgC)
    print(f"PSNR: {psnr_val}")
    # Calculate and print the SSIM value
    ssim_val = ssim(grayA, grayC)
    print(f"SSIM: {ssim_val}")
    # Calculate and print the MSE value
    mse_val = mse(grayA, grayC)
    print(f"MSE: {mse_val}")
    # Calculate and print the MAE value
    mae_val = mae(imgA, np.array(imgC))
    print(f"MAE: {mae_val}")

    print("------- Bicubic Metrics -------")
    # Calculate and print the PSNR value
    psnr_val = psnr(imgA, imgD)
    print(f"PSNR: {psnr_val}")
    # Calculate and print the SSIM value
    ssim_val = ssim(grayA, grayD)
    print(f"SSIM: {ssim_val}")
    # Calculate and print the MSE value
    mse_val = mse(grayA, grayD)
    print(f"MSE: {mse_val}")
    # Calculate and print the MAE value
    mae_val = mae(imgA, np.array(imgD))
    print(f"MAE: {mae_val}")

    fig, axs = plt.subplots(1, 4)
    fig.suptitle('Comparison', fontsize=16)

    axs[0].set_title("Ground Truth")
    axs[0].imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))

    axs[1].set_title("GANs")
    axs[1].imshow(cv2.cvtColor(np.array(imgB), cv2.COLOR_BGR2RGB))

    axs[2].set_title("Bicubic")
    axs[2].imshow(cv2.cvtColor(np.array(imgD), cv2.COLOR_BGR2RGB))

    axs[3].set_title("Bilinear")
    axs[3].imshow(cv2.cvtColor(np.array(imgC), cv2.COLOR_BGR2RGB))

    plt.show()

if __name__ == "__main__":
    main("PhIREGAN/wind test/wind images/wind/HR/ua_3461_28.png", "PhIREGAN/wind test/gans images/gans_ua_3461_28.png")