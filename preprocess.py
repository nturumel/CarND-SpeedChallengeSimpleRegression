# we will read each frame of the video and generate a corresponding dense optical flow 
import moviepy
import cv2
import numpy as np
import sys


def process_flow(flow_data):
    # we will divide the op_flow into 10 consecutive regions
    # we will conduct average pooling operations on each region
    # we do this because we expect optical flow values of neighboring pixels to be similar
    div_num = 10
    parts = np.array_split(np.array(flow_data), div_num, axis = 1)
    result = []
    for part in parts:
        result.append(np.mean(part, axis=(0, 1)))
    result = np.array(result)

    # normalisation
    max_val =  np.array([max(result[:, 0]), max(result[:, 1])])
    min_value = np.array([min(result[:, 0]), min(result[:, 1])])
    #FIXME: simplify this expression
    result[:, 0] = (result[:, 0] - min_value[0]) / (max_val[0] - min_value[0])
    result[:, 1] = (result[:, 1] - min_value[1]) / (max_val[1] - min_value[1])
    
    return result


def generate_opflow(video_file):
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
        crop_img_prev = prev[215:400, 85:560]
        crop_img_next = next[215:400, 85:560]
    
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
        dir = r"./cropped_images/"
        filename = dir + r"cropped_frame_" + str(i) + r".jpg"
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
    generate_opflow(r"data/train.mp4")
    check()