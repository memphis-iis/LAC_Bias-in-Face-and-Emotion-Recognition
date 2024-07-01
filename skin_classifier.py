# Description: This script uses the average pixel color of an image to classify skin color on the Fitzpatrick scale. 
# We must use the Von Luschan scale instead of the Fitzpatrick scale because we have rgb values for the Von Luschan scale.

import os
import re
import numpy as np
import scipy.spatial.distance as distance
import PIL.Image as Image
import cv2
import csv
import sys
from scipy.ndimage import binary_dilation
import rembg
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimage

base_directory = '/media/rusty/Data2/UNGA/UNGA_78/'
wmv_directory = base_directory + 'wmv/'
png_directory = base_directory + 'png_wmv_sample/'
img_directory = base_directory + 'png_face_samples/'
csv_directory = base_directory + 'csv_von_luschan_scale/'

def print_persistent(prompt, message):
    print(prompt, end='', flush=True)
    print("\r", end='', flush=True)
    print(message)
    

def get_sample_image_array_from_wmv_file(wmv_file_path, output_path, sample_rate=0.1, frame_number=False):
    #returns one random frame from the video, returns the rgb array of the frame as a numpy array
    #saves the frame as an image to the output_path, with filename /output_path/ + filename + "sample_image.png"
    print("get_sample_image_array_from_wmv_file")
    #get the video capture object
    cap = cv2.VideoCapture(wmv_file_path)
    #get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #set the random number for the frame to get if frame_number is not set
    if not frame_number:
        random_frame_number = random.randint(0, num_frames)
    else:
        random_frame_number = frame_number
    #set the frame number to get
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    #get the frame
    ret, frame = cap.read()
    #get the wmv_file_name
    wmv_file_name = os.path.basename(wmv_file_path)
    #save the frame as an image as a png
    cv2.imwrite(png_directory + wmv_file_name + "_sample_image.png", frame)
    #return the frame as a numpy array
    return frame

def get_image_rgb_array(image_path):
    print("get_image_rgb_array")
    image = Image.open(image_path)
    image_rgb_array = np.array(image.convert('RGB'))
    #print some info about the image
    print("image size: ", image.size)
    print("image mode: ", image.mode)
    print("image format: ", image.format)
    #print the width and height of the image
    width, height = image.size
    print("width: ", width, "height: ", height)
    #get the unique rgb values in the image
    unique_pixels = np.unique(image_rgb_array, axis=0)
    print("unique_pixels: ", unique_pixels.shape)
    return image_rgb_array

def get_transparent_image_rgb_array(image_path):
    print("get_transparent_image_rgb_array")
    image = Image.open(image_path)
    image_rgb_array = np.array(image.convert('RGBA'))
    #print some info about the image
    print("image size: ", image.size)
    print("image mode: ", image.mode)
    print("image format: ", image.format)
    #print the width and height of the image
    width, height = image.size
    print("width: ", width, "height: ", height)
    return image_rgb_array

def save_csv_from_array(array, output_path):
    print("save_csv_from_array")
    # Save the array as a csv
    np.savetxt(output_path, array, delimiter=",", fmt='%d')

def get_median_pixel_color(image_rgb_array):
    print("get_median_pixel_color")

    #flatten the image_rgb_array from a 3d array to a np list of rgb values
    image_rgb_array_flat = image_rgb_array.reshape(-1, image_rgb_array.shape[-1])

    #remove green pixels from the image_rgb_array7
    image_rgb_array_flat = image_rgb_array_flat[~np.all(image_rgb_array_flat == [0, 255, 0], axis=-1)]

    # Calculate the median pixel color
    median_pixel_color = np.median(image_rgb_array_flat, axis=0).astype(np.uint8)
    print("median_pixel_color: ", median_pixel_color)

    return median_pixel_color


def remove_non_skin_tone_colors_from_image(image_rgb_array, von_luschan_scale, filename=False, exclude_scale=[]):

    #create a mask for the image_rgb_array
    mask = np.zeros(image_rgb_array.shape[:2], dtype=np.uint8)


    #replace all colors in the image_rgb_array that are not in any of the von_luschan_scale arrays with green
    for key, value in von_luschan_scale.items():
        #if exclude_scale is set, skip the scale if its in the exclude_scale list
        if key in exclude_scale:
            continue
        mask[np.isin(image_rgb_array, value).all(axis=-1)] = 1

    #replace all colors in the image_rgb_array that are not in the dilated_mask with green
    image_rgb_array[mask == 0] = [0, 255, 0]
    
    #if file name is set, save the image_rgb_array as a png
    if filename:
        print("saving image_rgb_array as a png", filename)
        #remove the path and extension from the filename
        filename = os.path.splitext(os.path.basename(filename))[0]
        save_rgb_array_as_image(image_rgb_array, filename + "_vl_mask.png")
        save_rgb_array_as_image(mask * 255, filename + "_vlscale_only.png")

    
    return image_rgb_array



def get_array_of_colors_between_two_rgb_values(rgb1, rgb2):
    print("get_array_of_colors_between_two_rgb_values")
    min_r, max_r = min(rgb1[0], rgb2[0]), max(rgb1[0], rgb2[0])
    min_g, max_g = min(rgb1[1], rgb2[1]), max(rgb1[1], rgb2[1])
    min_b, max_b = min(rgb1[2], rgb2[2]), max(rgb1[2], rgb2[2])

    # get the amount of cells in the color_range
    cells = (max_r - min_r) * (max_g - min_g) * (max_b - min_b)

    # create a 1d array of zeros with the shape of the number of colors between each pair of rgb values
    color_array = np.zeros((int(cells), 3), dtype=np.uint8)

    # get items in color_array
    for i in range(color_array.shape[0]):
        # get the rgb values for each color in color_array
        color_array[i, 0] = min_r + i // ((max_g - min_g) * (max_b - min_b))
        color_array[i, 1] = min_g + (i // (max_b - min_b)) % (max_g - min_g)
        color_array[i, 2] = min_b + i % (max_b - min_b)

    return color_array


def get_closest_color_inside_array(target_color, color_array):
    # Convert the target_color to a 1D NumPy array, if it is a dict or list
    target_color = np.array(target_color)

    #flatten the color_array from a dict of numpy arrays to a 2d array of rgb values
    color_array_values = np.concatenate(list(color_array.values()))

    # Find the distance between the target_color and each color in the color_array
    distances = np.linalg.norm(color_array_values - target_color, axis=1)

    # Find the index of the color with the minimum distance of the rgb values
    closest_index = np.argmin(distances)

    # Get the closest color inside the color_array
    closest_color = color_array_values[closest_index]
    print("closest_color: ", closest_color)

    return closest_color


def get_x_y_index_of_color_or_closest_color_inside_array(color, color_array):
    print("get_x_y_index_of_color_or_closest_color_inside_array")
    #print array info
    print("color_array shape: ", color_array.shape)
    #if the color_array is an image of rgba values, remove the alpha channel
    if color_array.shape[2] == 4:
        color_array = color_array[:, :, :3]
    color = np.array(color)
    flat_color_array = color_array.reshape(-1, color_array.shape[-1])
    # Calculate the Euclidean distances for all colors in color_array
    distances = np.linalg.norm(flat_color_array - color, axis=1)
    # Find the index of the color with the minimum distance
    closest_index = np.argmin(distances)
    #get the x and y index of the closest color
    x_index = closest_index // color_array.shape[1]
    y_index = closest_index % color_array.shape[1]
    print("x_index: ", x_index, "y_index: ", y_index)
    return x_index, y_index



def get_von_lucian_scale_index_from_color(color, von_luschan_scale, dontIncludeScales=[]):
    print("get_von_lucian_scale_index_from_color", color)
    #iterate over all von_luschan_scale items and find the index of the closest color to the color
    final_index = 0
    final_color = np.array([0, 0, 0])
    min_color_dist = 1000000
    for key, value in von_luschan_scale.items():
        #if dontIncludeScales is set, skip the scale if its in the dontIncludeScales list
        if key in dontIncludeScales:
            continue
        #if there are no colors in the scale, it might be a tuple of rgb values
        if type(value) == tuple:
            value = np.array(value)
            mean_color = value
        else:            
            #get the mean color of the scale
            mean_color = np.mean(value, axis=0)
        #get the distance between the color and the mean color
        color_dist = distance.euclidean(color, mean_color)
        #if the color_dist is less than the min_color_dist, set the final_index to the key and the final_color to the mean_color
        if color_dist < min_color_dist:
            final_index = key
            final_color = mean_color
            min_color_dist = color_dist
        print("scale: ", key, "mean_color: ", mean_color, "color_dist: ", color_dist)
    #print the final_index, final_color, and the distances between the color and the final_color
    print("final_index: ", final_index, "final_color: ", final_color, "min_color_dist: ", min_color_dist)
    #wait for user input
    return final_index

def get_von_lucian_scale_index_from_lab_color(color, von_luschan_scale, dontIncludeScales=[]):
    print("get_von_lucian_scale_index_from_color", color)
    #iterate over all von_luschan_scale items and find the index of the closest color to the color
    final_index = 0
    final_color = np.array([0, 0, 0])
    min_color_dist = 1000000
    #convert the color to the lab color space
    color = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    #convert the von_luschan_scale to the lab color space
    for key, value in von_luschan_scale.items():
        #if dontIncludeScales is set, skip the scale if its in the dontIncludeScales list
        if key in dontIncludeScales:
            continue
        #if there are no colors in the scale, it might be a tuple of rgb values
        if type(value) == tuple:
            value = np.array(value)
            mean_color = value
        else:            
            #get the mean color of the scale
            mean_color = np.mean(value, axis=0)
        #convert the mean_color to the lab color space
        mean_color = cv2.cvtColor(np.array([[mean_color]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
        #get the distance between the color and the mean color
        color_dist = distance.euclidean(color, mean_color)
        #if the color_dist is less than the min_color_dist, set the final_index to the key and the final_color to the mean_color
        if color_dist < min_color_dist:
            final_index = key
            final_color = mean_color
            min_color_dist = color_dist
        print("scale: ", key, "mean_color: ", mean_color, "color_dist: ", color_dist)
    #print the final_index, final_color, and the distances between the color and the final_color
    print("final_index: ", final_index, "final_color: ", final_color, "min_color_dist: ", min_color_dist)
    #wait for user inputf
    return final_index




def convert_von_luschan_scale_to_fitzpatrick_scale(von_luschan_scale_index):
    print("convert_von_luschan_scale_to_fitzpatrick_scale")
    # Define the Fitzpatrick scale
    fitzpatrick_scale = {
        'I': (1, 5),
        'II': (6, 10),
        'III': (11, 15),
        'IV': (16, 21),
        'V': (22, 28),
        'VI': (29, 36)
    }
    # Convert the Von Luschan scale to the Fitzpatrick scale
    for key, value in fitzpatrick_scale.items():
        if von_luschan_scale_index in range(value[0], value[1] + 1):
            return key  
        

def get_array_of_color_distance(color_array, color, save=False):
    print("get_array_of_color_distance")
    # Get the array of color distances between color_array and color
    color_distance_array = np.abs(color_array - color)

    if save:
        # Scale the color_distance_array and save it as an image
        scaled_array = scale_rgb_array_to_max_distance(color_distance_array, 255)
        save_rgb_array_as_image(scaled_array, "color_distance_array.png")
    
    return color_distance_array



def remove_background_from_one_frame_of_video(video_file_path, output_path, exclude_scale=[], tolerances=[100, 75], frame_number=False, method="tol"):
    print("remove_background_from_one_frame_of_video", "tolerances: ", tolerances)
    #get the video capture object
    cap = cv2.VideoCapture(video_file_path)
    #get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #set the random number for the frame to get if frame_number is not set
    if not frame_number:
        random_frame_number = random.randint(0, num_frames)
    else:
        random_frame_number = frame_number
    #set the frame number to get
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    #get the frame
    ret, frame = cap.read()
    #get the wmv_file_name
    wmv_file_name = os.path.basename(video_file_path)
    #save the frame as an image as a png
    cv2.imwrite(png_directory + wmv_file_name + "_sample_image.png", frame)
    #remove the background from the image
    output = rembg.remove(frame)
    #strip filename of path and extension
    wmv_file_name = os.path.splitext(os.path.basename(video_file_path))[0]
    print("wmv_file_name: ", wmv_file_name)
    #save the output as a png
    cv2.imwrite(output_path + "_sample_image_no_background_" + str(random_frame_number) + ".png", output)
    #if the method is tol, remove the background from the image using the tolerance values
    if method == "tol":      
        #remove any colors within a tolerance of white or black
        white_balance_removal_tolerance = tolerances[0]
        #use np and distance.euclidean to remove any colors in the output that are not within the tolerance of white or black
        output = np.where(np.all(np.abs(output - [255, 255, 255, 255]) < white_balance_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        output = np.where(np.all(np.abs(output - [0, 0, 0, 255]) < white_balance_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        #remove tolerances from pure hues
        pure_hue_removal_tolerance = tolerances[1]
        #use np and distance.euclidean to remove any colors in the output that are not within the tolerance of pure hues
        output = np.where(np.all(np.abs(output - [255, 0, 0, 255]) < pure_hue_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        output = np.where(np.all(np.abs(output - [0, 255, 0, 255]) < pure_hue_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        output = np.where(np.all(np.abs(output - [0, 0, 255, 255]) < pure_hue_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        #remove any colors in the output that are not in the von_luschan_scale using np projections
        output = np.where(np.isin(output, von_luschan_scale_flat).all(axis=-1, keepdims=True), output, [0, 255, 0, 255])
        cv2.imwrite(output_path + "_vl_mask_whitebalance_" + str(white_balance_removal_tolerance) + "_purehues_" + str(pure_hue_removal_tolerance) + ".png", output)
        #remove alpha channel from the output
        rem_alpha = output[:, :, :3]
        non_green_pixels = np.count_nonzero(np.all(rem_alpha == [0, 255, 0], axis=-1))
        #remove all green pixels from the output
        output = output[~np.all(output == [0, 255, 0, 255], axis=-1)]
        #get the median color of the output
        median_color = np.median(output, axis=0).astype(np.uint8)
        print("median_color: ", median_color)
        #drop the alpha channel
        #wait for user input
        median_color = median_color[:3]
    #if method is "lab", remove the background from the image using the lab color space
    elif method == "lab":
        #remove the alpha channel from the output
        print("output shape: ", output.shape)
        #convert the output to the lab color space
        output_lab = skimage.color.rgb2lab(output[:, :, :3])
        #print the output_lab shape
        print("output_lab shape: ", output_lab.shape)
        #flatten the output_lab from a 3d array to 2d array
        output_lab_flat = output_lab.reshape(-1, output_lab.shape[-1])
        #convert green to lab color space
        green_lab = skimage.color.rgb2lab(np.array([[0, 255, 0]], dtype=np.uint8))
        print("green_lab: ", green_lab)
        #remove the green pixels from the output_lab
        output_lab_flat = output_lab_flat[~np.all(output_lab_flat == green_lab, axis=-1)]
        #remove 0, 0, 0 pixels from the output_lab
        output_lab_flat = output_lab_flat[~np.all(output_lab_flat == [0, 0, 0], axis=-1)]
        #get the median color of the output_lab
        median_color = np.median(output_lab_flat, axis=0).astype(np.uint8)
        print("median_color: ", median_color)
        #save the output as a png
        cv2.imwrite(output_path + filename + "_lab.png", output)
        non_green_pixels = np.count_nonzero(np.all(output_lab_flat == green_lab, axis=-1))
    elif method == "labtol":
        #remove any colors within a tolerance of white or black
        white_balance_removal_tolerance = tolerances[0]
        #use np and distance.euclidean to remove any colors in the output that are not within the tolerance of white or black
        output = np.where(np.all(np.abs(output - [255, 255, 255, 255]) < white_balance_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        output = np.where(np.all(np.abs(output - [0, 0, 0, 255]) < white_balance_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        #remove tolerances from pure hues
        pure_hue_removal_tolerance = tolerances[1]
        #use np and distance.euclidean to remove any colors in the output that are not within the tolerance of pure hues
        output = np.where(np.all(np.abs(output - [255, 0, 0, 255]) < pure_hue_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        output = np.where(np.all(np.abs(output - [0, 255, 0, 255]) < pure_hue_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        output = np.where(np.all(np.abs(output - [0, 0, 255, 255]) < pure_hue_removal_tolerance, axis=-1, keepdims=True), [0, 255, 0, 255], output)
        #get removed pixel count
        non_green_pixels = np.count_nonzero(np.all(output == [0, 255, 0, 255], axis=-1))
        #remove the alpha channel from the output
        print("output shape: ", output.shape)
        #convert the output to the lab color space
        output_lab = skimage.color.rgb2lab(output[:, :, :3])
        #print the output_lab shape
        print("output_lab shape: ", output_lab.shape)
        #flatten the output_lab from a 3d array to 2d array
        output_lab_flat = output_lab.reshape(-1, output_lab.shape[-1])
        #remove 0, 0, 0 pixels from the output_lab
        output_lab_flat = output_lab_flat[~np.all(output_lab_flat == [0, 0, 0], axis=-1)]
        #print how many 0, 0, 0 pixels were removed
        print("0, 0, 0 pixels removed: ", np.count_nonzero(np.all(output_lab_flat == [0, 0, 0], axis=-1)))
        #get the median color of the output_lab
        median_color = np.median(output_lab_flat, axis=0).astype(np.uint8)
        print("median_color: ", median_color)
        #save the output as a png
        cv2.imwrite(output_path + filename + "_labtol.png", output)
    return output, median_color, non_green_pixels

def remove_background_from_image(image_file_path, output_path, exclude_scale=[], tolerances=[100, 75], method="tol"):
    print("remove_background_from_image", "tolerances: ", tolerances)
    #open the image with cv2
    image = Image.open(image_file_path)
    #convert the image to a numpy array
    output = np.array(image)
    #remove alpha channel from the output
    rem_alpha = output[:, :, :3]
    non_green_pixels = np.count_nonzero(np.all(rem_alpha == [0, 255, 0], axis=-1))
    #remove all green pixels from the output
    output = output[~np.all(output == [0, 255, 0, 255], axis=-1)]
    #get the mean, median, and mode of the output
    mean_color = np.mean(output, axis=0).astype(np.uint8)
    median_color = np.median(output, axis=0).astype(np.uint8)
    mode_color = np.array([0, 0, 0])
    std_color = np.std(output, axis=0).astype(np.uint8)
    #remove colors outside of 3x std from the mean
    output = output[~np.all(np.abs(output - mean_color) > 3 * std_color, axis=-1)]
    #get the mean, median, and mode of the output
    mean_color = np.mean(output, axis=0).astype(np.uint8)
    median_color = np.median(output, axis=0).astype(np.uint8)
    mode_color = np.array([0, 0, 0])
    std_color = np.std(output, axis=0).astype(np.uint8)
    #drop the alpha channel
    #wait for user input
    median_color = median_color[:3]
    return output, median_color, non_green_pixels

                
def get_max_distance_betwewen_a_color_and_an_array_of_colors(color, color_array):
    print("get_max_distance_betwewen_a_color_and_an_array_of_colors")
    # Get the max distance between a color and an array of colors
    cells = color_array.shape[0] * color_array.shape[1]
    max_distance = 0
    for i in range(color_array.shape[0]):
        for j in range(color_array.shape[1]):
            color_distance = distance.euclidean(color, color_array[i, j])
            if color_distance > max_distance:
                max_distance = color_distance
                

    return max_distance

def scale_rgb_array_to_max_distance(rgb_array, max_distance):
    print("scale_rgb_array_to_max_distance", max_distance)
    # Scale rgb_array to max color distance
    rgb_array_scaled = np.zeros_like(rgb_array, dtype=np.uint8)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            for k in range(rgb_array.shape[2]):
                rgb_array_scaled[i, j, k] = rgb_array[i, j, k] * 255 / max_distance

    return rgb_array_scaled

def scale_rgb_array_count_to_another(rgb_array1, rgb_array2):
    print("scale_rgb_array_count_to_another")
    # Scale rgb_array1's indecies to rgb_array2's indecies
    rgb_array_scaled = np.zeros_like(rgb_array2, dtype=np.uint8)
    scale_x = rgb_array1.shape[0] / rgb_array2.shape[0]
    scale_y = rgb_array1.shape[1] / rgb_array2.shape[1]
    for i in range(rgb_array2.shape[0]):
        for j in range(rgb_array2.shape[1]):
            #get the nearest whole number index in rgb_array1
            index_x = int(i * scale_x)
            index_y = int(j * scale_y)
            rgb_array_scaled[i, j] = rgb_array1[index_x, index_y]

    return rgb_array_scaled
    
def multiply_rgb_array_by_another(rgb_array1, rgb_array2):
    print("multiply_rgb_array_by_another")
    # Multiply two rgb arrays
    rgb_array_product = np.zeros_like(rgb_array1, dtype=np.uint8)
    for i in range(rgb_array1.shape[0]):
        for j in range(rgb_array1.shape[1]):
            for k in range(rgb_array1.shape[2]):
                rgb_array_product[i, j, k] = rgb_array1[i, j, k] * rgb_array2[i, j, k]

    return rgb_array_product

def save_rgb_array_as_image(rgb_array, output_path):
    print("save_rgb_array_as_image")
    image = Image.fromarray(rgb_array)
    image.save(output_path)

def save_array_as_pickle(array, output_path):
    print("save_array_as_pickle")
    import pickle

    # Save the array as a pickle
    with open(output_path, "wb") as f:
        pickle.dump(array, f)

def load_array_from_pickle(input_path):
    print("load_array_from_pickle")
    import pickle

    # Load the array from a pickle
    with open(input_path, "rb") as f:
        array = pickle.load(f)

    return array

def check_if_file_exists(file_path):
    print("check_if_file_exists")
    # Check if the file exists
    return os.path.exists(file_path)


def define_von_luschan_scale(load=False):
    print("define_von_luschan_scale")
    #if the file exists, load the von_luschan_scale from the csv
    if load:
        if check_if_file_exists(load):
            von_luschan_scale = load_array_from_pickle(load)
            return von_luschan_scale
            
    # Define the Von Luschan scale
    rgb_values = [
        (244,242,245),
        (237,236,234),
        (250,249,247),
        (253,251,230),
        (253,246,230),
        (254,247,229),
        (250,240,239),
        (243,234,229),
        (244,241,234),
        (251,252,244),
        (250,240,240),
        (243,235,230),
        (244,241,235),
        (251,252,244),
        (241,231,195),
        (252,248,238),
        (255,249,226),
        (241,232,196),
        (240,227,175),
        (242,227,153),
        (236,215,160),
        (236,218,134),
        (223,193,123),
        (228,198,124),
        (226,194,106),
        (224,194,124),
        (223,185,120),
        (189,152,98),
        (157,107,65),
        (122,76,44),
        (101,48,32),
        (100,45,14),
        (101,44,26),
        (96,45,27),
        (86,46,36),
        (62,26,13),
        (45,32,36),
        (20,22,42)






    ]
    # get the rgb values for the colors between each pair of rgb values in the Von Luschan scale
    #crate numpy array of zeros with the shape of the number of colors between each pair of rgb values
    von_luschan_scale = {}
    for i in range(len(rgb_values) - 1):
        von_luschan_scale[i] = get_array_of_colors_between_two_rgb_values(rgb_values[i], rgb_values[i + 1])
        
    #if load is set, save the von_luschan_scale to a csv
    if load:
        save_array_as_pickle(von_luschan_scale, load)
        print("saved von_luschan_scale to pickle")
        #pretty print the von_luschan_scale
        print("von_luschan_scale: ", von_luschan_scale)

    #create a dictionary of numpy arrays of rgb values
    van_luscian_source_values = {}
    for i in range(len(rgb_values) - 1):
        van_luscian_source_values[i] = rgb_values[i]
    #return the von_luschan_scale and the rgb_values
    print("van_luscian_source_values: ", van_luscian_source_values)
    print("von_luschan_scale: ", von_luschan_scale)
    return von_luschan_scale, van_luscian_source_values

def flatten_van_luschan_scale(load=None, von_luschan_scale=None):
    print("flatten_van_luschan_scale")
    
    # If von_luschan_scale is a file, load it instead of flattening it
    if isinstance(load, str):
        if check_if_file_exists(load):
            von_luschan_scale = load_array_from_pickle(load)
            return von_luschan_scale
        #we reshape the von_luschan_scale to a 2d array of rgb values. it currently is a 3d array of rgb values
        print("reshaping von_luschan_scale")
        #its a dictionary of numpy arrays, so we need to iterate through the dictionary and combine the numpy arrays
        von_luschan_scale_flat = np.concatenate(list(von_luschan_scale.values()))
        #save the von_luschan_scale_flat as a csv
        save_array_as_pickle(von_luschan_scale_flat, load)
        save_csv_from_array(von_luschan_scale_flat, load + ".csv")
        print("saved von_luschan_scale_flat to pickle")
        return von_luschan_scale_flat


    #convert the list to a numpy array
    valid_arrays = np.array(valid_arrays)

    #if load is set, save the von_luschan_scale to a csv
    if isinstance(load, str):
        save_array_as_pickle(valid_arrays, load)
        print("saved von_luschan_scale to pickle")

    return valid_arrays               
        


def get_color_array_for_fitzpatrick_scale(fitzpatrick_scale_index):
    print("get_color_array_for_fitzpatrick_scale")
    # Define the Fitzpatrick scale
    fitzpatrick_scale = {
        'I': (1, 5),
        'II': (6, 10),
        'III': (11, 15),
        'IV': (16, 21),
        'V': (22, 28),
        'VI': (29, 36)
    }
    # Define the Von Luschan scale
    von_luschan_scale = define_von_luschan_scale()
    # Get the color array for the Fitzpatrick scale
    color_array = {}
    for key, value in fitzpatrick_scale.items():
        if key == fitzpatrick_scale_index:
            for i in range(value[0], value[1] + 1):
                color_array[i] = von_luschan_scale[i]
    
    return color_array

def convert_rgb_to_grey(r, g, b):
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    return [grey, grey, grey]

def get_bounding_box_with_flat_von_luscian(color_array, median_color_position, von_luschan_scale_flat, tolerance=10, exclude_color=[0, 255, 0]):
    rows, cols, _ = color_array.shape

    center_x, center_y = median_color_position

    # Initialize the bounding box
    bound_top_left_x = center_x
    bound_top_left_y = center_y
    bound_bottom_right_x = center_x
    bound_bottom_right_y = center_y

    # Create a mask for the background pixels
    exclude_colors_mask = np.all(np.isin(color_array, exclude_color), axis=-1)

    # create a mask for the von_luschan_scale_flat pixels
    von_luschan_scale_mask = np.isin(color_array, von_luschan_scale_flat).all(axis=-1)

    #combine the exclude_colors_mask and von_luschan_scale_mask
    valid_pixels_mask = ~(exclude_colors_mask | von_luschan_scale_mask)

    #save the valid_pixels_mask as an image
    save_rgb_array_as_image(valid_pixels_mask, "valid_pixels_mask.png")

    #dilate the valid_pixels_mask to the tolerance
    dilated_valid_pixels_mask = binary_dilation(valid_pixels_mask, structure=np.ones((3, 3)), iterations=tolerance)

    #find the bounding box of the dilated_valid_pixels_mask where the mask value is true
    for i in range(rows):
        for j in range(cols):
            if dilated_valid_pixels_mask[i, j]:
                if i < bound_top_left_x:
                    bound_top_left_x = i
                if i > bound_bottom_right_x:
                    bound_bottom_right_x = i
                if j < bound_top_left_y:
                    bound_top_left_y = j
                if j > bound_bottom_right_y:
                    bound_bottom_right_y = j

    #print the percentage of colors in the array that are in the von_luschan_scale_flat (where the mask value is true)
    print("percentage of colors in the array that are in the von_luschan_scale_flat: ", np.count_nonzero(von_luschan_scale_mask) / von_luschan_scale_mask.size * 100, "%")
    
    return bound_top_left_x, bound_bottom_right_x, bound_top_left_y, bound_bottom_right_y, center_x, center_y




def draw_bounding_box_on_image(image_rgb_array, x1, x2, y1, y2, x_center=False, y_center=False):
    print("draw_bounding_box_on_image")

    # Ensure bounding box coordinates are within the image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_rgb_array.shape[0] - 1, x2)
    y2 = min(image_rgb_array.shape[1] - 1, y2)

    # Draw the bounding box on the image
    image_rgb_array_bounding_box = np.copy(image_rgb_array)
    image_rgb_array_bounding_box[x1:x2 + 1, y1:y2 + 1] = image_rgb_array[x1:x2 + 1, y1:y2 + 1]

    # Draw the white border
    border_thickness = 1
    image_rgb_array_bounding_box[x1:x1 + border_thickness, y1:y2 + 1] = [255, 255, 255]  # Top border
    image_rgb_array_bounding_box[x2:x2 + border_thickness + 1, y1:y2 + 1] = [255, 255, 255]  # Bottom border
    image_rgb_array_bounding_box[x1:x2 + 1, y1:y1 + border_thickness] = [255, 255, 255]  # Left border
    image_rgb_array_bounding_box[x1:x2 + 1, y2:y2 + border_thickness + 1] = [255, 255, 255]  # Right border

    # Label corners
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Adjusted font size
    font_thickness = 1

    cv2.putText(image_rgb_array_bounding_box, f"({x1}, {y1})", (y1, x1), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(image_rgb_array_bounding_box, f"({x1}, {y2})", (y2, x1), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(image_rgb_array_bounding_box, f"({x2}, {y1})", (y1, x2), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(image_rgb_array_bounding_box, f"({x2}, {y2})", (y2, x2), font, font_scale, (255, 255, 255), font_thickness)

    #draw a cross at the x_center and y_center
    cv2.line(image_rgb_array_bounding_box, (y_center - 10, x_center), (y_center + 10, x_center), (255, 255, 255), 1)
    cv2.line(image_rgb_array_bounding_box, (y_center, x_center - 10), (y_center, x_center + 10), (255, 255, 255), 1)


    return image_rgb_array_bounding_box

def draw_bounding_boxes_on_image(image_rgb_array, bounding_boxes):
    image_rgb_array_bounding_boxes = np.copy(image_rgb_array)

    for x1, x2, y1, y2, x_center, y_center in bounding_boxes:
        image_rgb_array_bounding_boxes = draw_bounding_box_on_image(image_rgb_array_bounding_boxes, x1, x2, y1, y2, x_center, y_center)

    return image_rgb_array_bounding_boxes

def replace_colors_with_closest_von_luschan_color(image_rgb_array, von_luschan_scale_flat, tolerance=100, chunk_size=1000):
    print("replace_colors_with_closest_von_luschan_color")

    #save the von_luschan_scale_flat as a csv
    save_csv_from_array(von_luschan_scale_flat, "von_luschan_scale_flat.csv")

    height, width, channels = image_rgb_array.shape
    print("height: ", height, "width: ", width, "channels: ", channels)

    # Reshape the image to a 2D array of pixels
    print("reshaping image_rgb_array")
    pixels = np.array(image_rgb_array.reshape((-1, channels)))

    # Calculate the number of chunks
    print("calculating the number of chunks")
    num_chunks = int(np.ceil(len(pixels) / chunk_size))
    print("num_chunks: ", num_chunks)

    # Process the image in chunks
    for i in range(num_chunks):
        print("processing chunk: ", i)
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pixels))

        # Extract the chunk of pixels
        chunk_pixels = pixels[start_idx:end_idx]

        # Calculate the Euclidean distances for the chunk
        distances = np.linalg.norm(von_luschan_scale_flat - chunk_pixels[:, np.newaxis], axis=-1)

        # Find the index of the color with the minimum distance for the chunk
        closest_indices = np.argmin(distances, axis=1)

        # Use the indices to get the closest colors from von_luschan_scale_flat for the chunk
        closest_colors = von_luschan_scale_flat[closest_indices]

        # Replace the corresponding pixels in the chunk
        pixels[start_idx:end_idx] = closest_colors

        #print a index of the chunk and it's resulting closest_colors
        print("chunk_color_at_index_0: ", chunk_pixels[0], "closest_color_at_index_0: ", closest_colors[0])

        #check if closest_colors[0] is in von_luschan_scale_flat
        if not np.isin(closest_colors[0], von_luschan_scale_flat).all(axis=-1):
            print("closest_colors[0] is not in von_luschan_scale_flat")
        

    # Reshape the resulting array back to the original image shape
    image_rgb_array_replaced = pixels.reshape((height, width, channels))

    return image_rgb_array_replaced

def replace_colors_with_closest_von_luschan_color_from_index(image_rgb_array, von_luschan_scale, color_index, tolerance=100, chunk_size=1000):
    print("replace_colors_with_closest_von_luschan_color_from_index")
    print("image_rgb_array shape: ", image_rgb_array.shape)
    height, width, channels = np.array(image_rgb_array).shape
    

    # Reshape the image to a 2D array of pixels using np
    print("reshaping image_rgb_array")
    pixels = np.array(image_rgb_array.reshape((-1, channels)))

    # Get the color array for the specified index
    print("getting the color_array for the specified index")
    color_array = von_luschan_scale[color_index]

    #print some info about the color_array
    print("color_array shape: ", color_array.shape)

    # Calculate the number of chunks
    num_chunks = int(np.ceil(len(pixels) / chunk_size))

    # Process the image in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pixels))

        # Extract the chunk of pixels
        chunk_pixels = pixels[start_idx:end_idx]

        # Calculate the Euclidean distances for the chunk
        distances = np.linalg.norm(color_array - chunk_pixels[:, np.newaxis], axis=-1)

        # Find the index of the color with the minimum distance for the chunk
        closest_indices = np.argmin(distances, axis=1)

        # Use the indices to get the closest colors from the color_array for the chunk
        closest_colors = color_array[closest_indices]

        # Replace the corresponding pixels in the chunk
        pixels[start_idx:end_idx] = closest_colors

    # Reshape the resulting array back to the original image shape
    image_rgb_array_replaced = pixels.reshape((height, width, channels))

    return image_rgb_array_replaced


def replace_skin_tone_with_target_von_luscian_scale(original_image_array, von_luschan_scale, target_von_luschan_scale_index, tolerance=100, chunk_size=1000):
    print("replace_skin_tone_with_target_von_luscian_scale")
    #get the resolution of the image
    height, width, channels = original_image_array.shape
    #get all rgb values in the image
    pixels = original_image_array.reshape((-1, channels))
    #get a array of unique rgb values in the image
    unique_pixels = np.unique(pixels, axis=0)
    print("unique_pixels: ", unique_pixels.shape)
    #get the color array for the specified index
    target_color_array = von_luschan_scale[target_von_luschan_scale_index]
    #get the unique pixels that are in the target color array
    unique_target_pixels = np.unique(target_color_array, axis=0)
    print("unique_target_pixels: ", unique_target_pixels.shape)
    #we need to scale the unique_target_pixels to the same size as unique_pixels
    #get the number of unique pixels in the image
    num_unique_pixels = unique_pixels.shape[0]
    #get the number of unique pixels in the target color array
    num_unique_target_pixels = unique_target_pixels.shape[0]
    #get the scale factor
    scale_factor = num_unique_pixels / num_unique_target_pixels
    #scale the unique_target_pixels
    scaled_unique_target_pixels = scale_rgb_array_count_to_another(unique_pixels, unique_target_pixels)
    print("scaled_unique_target_pixels: ", scaled_unique_target_pixels.shape)
    #replace the unique pixels in the image with the scaled unique_target_pixels
    image_rgb_array_replaced = replace_colors_with_closest_von_luschan_color(original_image_array, scaled_unique_target_pixels, tolerance, chunk_size)
    return image_rgb_array_replaced


def shift_von_luschan_colors_in_array_to_new_median_color(image_rgb_array, von_luschan_scale_flat, target_median_color, tolerance=100, chunk_size=1000):
    print("shift_von_luschan_colors_in_array_to_new_median_color")
    #get the median color of the image_rgb_array
    median_color = get_median_pixel_color(image_rgb_array)
    print("median_color: ", median_color)
    #get the distance between the median color and the target median color 
    r_distance = target_median_color[0] - median_color[0]
    g_distance = target_median_color[1] - median_color[1]
    b_distance = target_median_color[2] - median_color[2]
    print("r_distance: ", r_distance, "g_distance: ", g_distance, "b_distance: ", b_distance)
    distances = np.array([r_distance, g_distance, b_distance])
    #we only want to shift the colors in the von_luschan_scale_flat
    #get the unique colors in the image_rgb_array
    unique_pixels = np.unique(image_rgb_array, axis=0)
    print("unique_pixels: ", unique_pixels.shape)
    #get the unique colors in the von_luschan_scale_flat
    unique_von_luschan_scale_flat = np.unique(von_luschan_scale_flat, axis=0)
    print("unique_von_luschan_scale_flat: ", unique_von_luschan_scale_flat.shape)
    #get the unique colors in the von_luschan_scale_flat that are also in the image_rgb_array
    unique_von_luschan_scale_flat_in_image = np.intersect1d(unique_von_luschan_scale_flat, unique_pixels)
    print("unique_von_luschan_scale_flat_in_image: ", unique_von_luschan_scale_flat_in_image.shape)
    #get the unique colors in the von_luschan_scale_flat that are not in the image_rgb_array
    unique_von_luschan_scale_flat_not_in_image = np.setdiff1d(unique_von_luschan_scale_flat, unique_pixels)
    print("unique_von_luschan_scale_flat_not_in_image: ", unique_von_luschan_scale_flat_not_in_image.shape)
    #shift the unique colors in the von_luschan_scale_flat that are also in the image_rgb_array using numpy broadcasting
    unique_von_luschan_scale_flat_in_image_shifted = unique_von_luschan_scale_flat_in_image + distances
    print("unique_von_luschan_scale_flat_in_image_shifted: ", unique_von_luschan_scale_flat_in_image_shifted.shape)
    #combine the unique_von_luschan_scale_flat_in_image_shifted and unique_von_luschan_scale_flat_not_in_image
    unique_von_luschan_scale_flat_shifted = np.concatenate((unique_von_luschan_scale_flat_in_image_shifted, unique_von_luschan_scale_flat_not_in_image))
    print("unique_von_luschan_scale_flat_shifted: ", unique_von_luschan_scale_flat_shifted.shape)
    #replace the unique pixels in the image with the scaled unique_target_pixels
    image_rgb_array_replaced = replace_colors_with_closest_von_luschan_color(image_rgb_array, unique_von_luschan_scale_flat_shifted, tolerance, chunk_size)
    return image_rgb_array_replaced


#set paths
von_luschan_scale_path = "./von_luschan_scale.pickle"
von_luschan_scale_flat_path = "./von_luschan_scale_flat.pickle"

#define the von luschan scale
von_luschan_scale, rgb_values = define_von_luschan_scale(von_luschan_scale_path)


#flatten the von luschan scale
von_luschan_scale_flat = flatten_van_luschan_scale(von_luschan_scale_flat_path, von_luschan_scale)

#get the files in the path
files = os.listdir(img_directory)

#print number of files
print("number of files: ", len(files))

#create a csv with the headers filename, von_luschan_scale_index
with open(csv_directory + "von_luschan_scale_index.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "von_luschan_scale_index", "fitzpatrick_scale_index", "median_sample_color_minux_3x_std_outliers"])

#iterate through the files
for file in files:
    #check if the file is a wmv
    if file.endswith(".png"):
        #get the file path
        file_path = os.path.join(img_directory, file)

        #get the first 3 letters of the filename
        filename = file.split(".")[0][:3]

        exclude = [0, 255, 0]


        tolerances = [100]
        tolerances2 = [75]
        #remove the background from the image
        for tolerance in tolerances:
            white_balance_removal_tolerance = tolerance
            for tolerance2 in tolerances2:
                pure_hue_removal_tolerance = tolerance2
                print("white_balance_removal_tolerance: ", white_balance_removal_tolerance, "pure_hue_removal_tolerance: ", pure_hue_removal_tolerance)
                image_rgb_array_removed, median_color, non_green_pixels = remove_background_from_image(file_path, exclude, [white_balance_removal_tolerance, pure_hue_removal_tolerance])

                #get percentage of non green pixels
                non_green_pixels_percentage = 0

                #pretty print the median_color
                print("median_color: ", median_color)

                #drop the alpha channel
                median_color = median_color[:3]
                

                #get the von_luschan_scale_index from the median_color
                von_luschan_scale_index = get_von_lucian_scale_index_from_color(median_color, rgb_values)

                #get the fitzpatrick_scale_index from the von_luschan_scale_index
                fitzpatrick_scale_index = convert_von_luschan_scale_to_fitzpatrick_scale(von_luschan_scale_index)

                #print the von_luschan_scale_index, and then write it to a file
                print("von_luschan_scale_index: ", von_luschan_scale_index)


                #write the filename and von_luschan_scale_index to a csv
                with open(csv_directory + "von_luschan_scale_index.csv", "a") as file:
                    writer = csv.writer(file)
                    writer.writerow([filename, von_luschan_scale_index, fitzpatrick_scale_index, median_color])
                    print("writing to csv")
                    print("filename: ", filename, "von_luschan_scale_index: ", von_luschan_scale_index, "fitzpatrick_scale_index: ", fitzpatrick_scale_index, "median_color: ", median_color, "white_balance_removal_tolerance: ", white_balance_removal_tolerance, "pure_hue_removal_tolerance: ", pure_hue_removal_tolerance, "non_green_pixels: ", non_green_pixels, "non_green_pixels_percentage: ", non_green_pixels_percentage, "method: tol")







            

            

          

    

            






                