#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import platform
import sys
import time
import numpy as np
import matplotlib
import yaml
import random
from numpy.random import default_rng

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# read configuration file
def read_yaml(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

# read iamge from input location
def read_image(image_file):
    with Image.open(image_file) as img:
        return np.array(img)

#Covert iamge to single channel 
def convert_image_to_single_channel(color_img, choice):


    if(choice == 'red'):
        return color_img[:,:,0]
    elif(choice == 'green'):
        return color_img[:,:,1]
    elif(choice == 'blue'):
        return color_img[:,:,2]
    
# Histogram calculation for each individual image
    
def plot_histogram(histogram, filename, applied_method):
    idk, bins = np.histogram(histogram, bins=256, range=(0, 256))
    plt.title("histogram")
    plt.figure()
    plt.plot(bins[0:-1], histogram)
    plt.savefig(safe_conf["OUTPUT_PATH"] + filename + applied_method + ".png")
    plt.close()

# https://stackoverflow.com/questions/21669657/getting-cannot-write-mode-p-as-jpeg-while-operating-on-jpg-image
def save_image(img, filename, applied_method):
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_PATH"] + filename + applied_method + ".jpg", format="JPEG")


# ceate histogram
def calc_histogram(img):

    
    histogram = np.zeros(256)
    img_size = len(img)
    
    for l in range(256):
      for i in range(img_size):
        if img.flat[i] == l:
            histogram[l] += 1

    return histogram



# Histogram equalization for each image
# used numpy cumsum() function that returns the cumulative sum of the elements along the given axis.
# first flattened array, calculate cumulative sum, normalize values between 0-256 and then flattend histogram

# https://www.analyticsvidhya.com/blog/2022/01/histogram-equalization/
 
def equalization(histogram, img):
   
    img_flattened = img.flatten()   
    cum_sum = np.cumsum(histogram)   
    normalization = (cum_sum - cum_sum.min()) * 255
    n = cum_sum.max() - cum_sum.min()   
    uniform_norm = normalization / n
    uni = uniform_norm.astype(np.uint8)
    
    
    image_eq = uni[img_flattened]
    image_eq = np.reshape(image_eq, img.shape)

    return calc_histogram(image_eq), image_eq

# Noise Addition : Salt and Pepper Noise
# first add salt 255, get number of pixels to salt and multiply with strength. loop through copy iamge  
# salt the values while picking random x and y coordinate. color pixel to white
# add pepper 0, get number of pixels to pepper and multiply with user specified strength
# loop through image and add pepper while picking random x and y coordinate. color pixel to black
def salt_pepper(img, strength):
    
    row, col = img.shape
    cons = 0.5
    
    num_pixels_white = random.randint(0, img.size)
    num_pixels_white = strength * num_pixels_white * cons

    img_copy = np.copy(img)

    for i in range(int(num_pixels_white)):

        x=random.randint(0, col - 1)
        y=random.randint(0, row - 1)
        img_copy[y][x] = 255

    num_pixels_black = random.randint(0, img.size)
    num_pixels_black = strength * num_pixels_black * (1 - cons)
    
# add pepper 
    for i in range(int(num_pixels_black)):
        x=random.randint(0, col - 1)
        y=random.randint(0, row - 1)
        img_copy[y][x] = 0
         
    return img_copy

# Noise Addition : Gaussian noise
# set mean to 0

def gaussian(img, strength):
    mean = 0.0

    row, col = img.shape
    rng = default_rng()
    noise = rng.normal(mean, strength, size=(row,col))

    noise_reshape = noise.reshape(img.shape)

    copy_img = img + noise_reshape
    
    return copy_img

# Filtering operations : linear filter
def linear_filter(img, weights):
    
      # 3 x 3
    kernel = np.array(weights) 
  
    rows, cols = img.shape 
    # 3 x 3
    mask_rows, mask_cols = kernel.shape 

    copy_img = np.zeros((rows, cols))

    for row in tqdm(range(1, rows - 1)):
        for col in range(1, cols - 1):
            for i in range(mask_rows):
                for j in range(mask_cols):
                    intensity = img[row - i + 1, col - j + 1]
                    average = int(np.average(intensity * kernel))
            copy_img[row, col] = average
    return copy_img

# Filtering operations : median filter

def median_filter(img, weights):
  
    kernel = np.array(weights) 

    rows, cols = img.shape 
    # 3 x 3
    kernel_rows, kernel_cols = kernel.shape 

    window = np.zeros(kernel.size) 

    copy_img = np.zeros((rows, cols)) 

    for row in tqdm(range(1, rows - 1)):
        for col in range(1, cols - 1): 
            pixel = 0
            for i in range(kernel_rows):
                for j in range(kernel_cols):
                    # store neighbor pixel values in window
                    window[pixel] = img[row - i + 1][col - j + 1]
                    pixel += 1

            window.sort()

            copy_img[row][col] = window[pixel // 2]
        
    copy_img = copy_img.astype(np.uint8)

    return copy_img

# Image Quantization:
def quantized(img, histogram, quant_size):

    lowest = np.min(histogram)
    highest = np.max(histogram)

    quartile = (highest - lowest) / quant_size
    blah = int(256 / quartile)

    copy_img = np.copy(img)
    copy_img = (np.floor_divide(img, quartile)).astype(np.uint8)
    copy_img = (copy_img * blah).astype(np.uint8)

    return copy_img

# Mean square error calculation
def mse(og_img, quantized_img):
    mserror = (np.square(np.subtract(og_img, quantized_img))).mean()
    
    return mserror
    
def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_PATH'])

    Path(safe_conf['OUTPUT_PATH']).mkdir(parents=True, exist_ok=True)
    
    files = list(data_loc.iterdir())
    filenames = [i for i in range(len(files))]
    r, c = 256, 7
    average_classes = [[0 for i in range(r)] for y in range(c)]
    num = [0 for i in range(c)]
    
    timings = []
    time_start = time.perf_counter()
    
    for i in range(len(files)):
        
        ts = time.perf_counter()
        
        filenames[i] = os.path.basename(files[i])
        
        if (".BMP" in filenames[i]):
            filenames[i] = os.path.splitext(filenames[i])[0]

        color_image = read_image(files[i])
        
        # convert img to greyscale / proper color channel
        print ("convert to greyscale")
        img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
        save_image(img, filenames[i], "_greyscale")
        
        # create histograms
        histogram = calc_histogram(img)
        plot_histogram(histogram, filenames[i], "_histogram")

        # sum of each hist for each class
        if("cyl" in filenames[i]):
            average_classes[0] += histogram
            num[0]+= 1
            
        elif("inter" in filenames[i]):
            average_classes[1] += histogram
            num[1]+= 1
            
        elif("let" in filenames[i]):
            average_classes[2] += histogram
            num[2]+= 1
            
        elif("mod" in filenames[i]):
            average_classes[3] += histogram
            num[3]+= 1
            
        elif("para" in filenames[i]):
            average_classes[4] += histogram
            num[4]+= 1
            
        elif("super" in filenames[i]):
            average_classes[5] += histogram
            num[5]+= 1
            
        elif("svar" in filenames[i]):
            average_classes[6] += histogram
            num[6]+= 1
        
        # add salt & pepper noise then save
        print("salt and pepper noise")
        snp_img = salt_pepper(img, safe_conf["SNP_NOISE"])
        save_image(snp_img, filenames[i], "_salt_pepper")
        print("Image with salt and pepper noise is saved")

        # add gaussian noise then save
        print("gaussian noise")
        gaussian_img = gaussian(img, safe_conf["G_NOISE"])
        save_image(gaussian_img, filenames[i], "_gaussian")
        print("Image with gaussian noise is saved")
        
        # create equalized histogram and equalized image
        print("equalization")
        equalized, image_eq = equalization(histogram, img)
        plot_histogram(equalized, filenames[i], "_equalized")
        save_image(image_eq, filenames[i], "_equalized")
        print("Image with histogram equalization is saved")

        
        # create quantized image
        print("quantization")
        quantized_img = quantized(img, histogram, 8)
        save_image(quantized_img, filenames[i], "_quantized")
        print("Image with quantization is saved")
        
        # apply linear filter to salted images
        print("linear filter")
        linear = linear_filter(img, safe_conf["LINEAR_WEIGHT"])
        save_image(linear, filenames[i], "_linear_filter")
        print("After applying linear filter, image is saved")
        
        # apply median filter to salted images
        print("median filter")
        median = median_filter(img, safe_conf["MEDIAN_WEIGHT"])
        save_image(median, filenames[i], "_median_filter")
        print("After applying median filter, image is saved")
        
        # calculate mean square error
        msqe = mse(img, image_eq)
        print("\n", filenames[i], "msqe: ", msqe, "\n")
        
        te = time.perf_counter()

        print("timing: ", te-ts, "\n")
    
    # Averaged histograms of pixel values for each class of images.
    for y in range(c):
        for x in range(r):
            average_classes[y][x] = int(average_classes[y][x] / num[y])
    
    plot_histogram(average_classes[0], "cyl", "_avg")
    plot_histogram(average_classes[1], "inter", "_avg")
    plot_histogram(average_classes[2], "let", "_avg")
    plot_histogram(average_classes[3], "mod", "_avg")
    plot_histogram(average_classes[4], "para", "_avg")
    plot_histogram(average_classes[5], "super", "_avg")
    plot_histogram(average_classes[6], "svar", "_avg")
 
    avg_time = (sum(timings) / len(timings))
    
    time_end = time.perf_counter()
    print("\nTotal execution time for single threaded processing: ", time_end - time_start)
    print("\naverage time for each image to process: ", avg_time)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




