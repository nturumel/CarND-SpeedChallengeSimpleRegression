from globals import global_var

import cv2
import numpy as np
import sys 


def process_flow(flow_data):
    # we will divide the op_flow into 10 consecutive regions
    # we will conduct average pooling operations on each region
    # we do this because we expect optical flow values of neighboring pixels to be similar
    parts = np.array_split(np.array(flow_data), global_var['DIV_NUM_X'], axis=1)
    result = []
    for part in parts:
        result.append(np.mean(part, axis=(0, 1)))
    result = np.array(result)

    '''
    # normalisation
    max_val =  np.array([max(result[:, 0]), max(result[:, 1])])
    min_value = np.array([min(result[:, 0]), min(result[:, 1])])
    result[:, 0] = (result[:, 0] - min_value[0]) / (max_val[0] - min_value[0])
    result[:, 1] = (result[:, 1] - min_value[1]) / (max_val[1] - min_value[1])
    '''
    return result


def generate_opflow():
    # initialsise
    video_file = global_var['TRAIN_FILE']
    
    # initialise empty list for results
    result =[]

    # start processing frame by frame
    video_reader = cv2.VideoCapture(video_file)
    ret, prev = video_reader.read()
    i = 0
    while(True):
        ret, next = video_reader.read()
        if not ret:
            break 

        # crop to frame 
        CROP_RANGE = global_var['CROP_RANGE']
        crop_img_prev = prev[CROP_RANGE[0]:CROP_RANGE[1], CROP_RANGE[2]:CROP_RANGE[3]]
        crop_img_next = next[CROP_RANGE[0]:CROP_RANGE[1], CROP_RANGE[2]:CROP_RANGE[3]]
    
        # convert to grey scale
        crop_img_prev = cv2.cvtColor(crop_img_prev, cv2.COLOR_BGR2GRAY)
        crop_img_next = cv2.cvtColor(crop_img_next, cv2.COLOR_BGR2GRAY)

        # prev is next
        prev = next

        # calculate dense optical flow
        flow_data = cv2.calcOpticalFlowFarneback(crop_img_prev, crop_img_next, None, 0.4, 1, 12, 2, 8, 1.2, 0)

        # process flow data
        result.append(process_flow(flow_data))

        # store cropped image for reference
        filename = CROP_RANGE['CROP_IMG'] + str(i) + r".jpg"
        cv2.imwrite(filename, crop_img_prev)
        i += 1
        sys.stdout.write('Opflow Processing frame number %d\r' % i)
        sys.stdout.flush()

    # save results    
    result = np.array(result)
    np.save("preprocessed_values", result)
    
def check(save_array_file = r"preprocessed_values.npy"):
    a = np.load(save_array_file)
    x, y, z = a.shape
    count = 0

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if a[i, j, k] > 1 or a[i, j, k] < 0:
                    print("problem, value = %f, indice = %d, %d, %d" % (a[i, j, k], i, j, k))

if __name__ == '__main__':
    generate_opflow()
    check()