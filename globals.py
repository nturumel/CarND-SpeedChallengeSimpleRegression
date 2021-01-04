global_var = {}

def initialise():
    global_var['DIV_NUM_X'] = 10
    global_var['DIV_NUM_Y'] = 1
    global_var['CROP_IMG'] = r"./cropped_images/" +  r"cropped_frame_"
    global_var['CROP_RANGE'] = [215, 400, 85, 560]
    global_var['SAVE_ARRAY_FILE'] = r"data/preprocessed_values.npy"
    global_var['TRAIN_OUTPUT'] = r"data/train.txt" 
    global_var['PREDICTIONS_OUTPUT'] = r"data/predictions.txt" 
    global_var['TRAIN_FILE'] = r"data/train.mp4"
    global_var['TEST_FILE'] = r"data/test.mp4"
    global_var['INPUT_SHAPE'] = [global_var['DIV_NUM_X'] * global_var['DIV_NUM_Y'], 2]
    global_var['MODEL_PATH'] = r"speed_modules/speed_predictor_model"
    global_var['LOG_DIR'] = r'./speed_modules/logs/'
    global_var['BATCH_SIZE'] = 256
    global_var['EPOCHS'] = 50
    global_var["SCALE_OUTPUT"] = 100

def set_var(variable, value):
    variable = str(variable)
    global_var[variable] = value

