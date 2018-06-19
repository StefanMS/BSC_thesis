import cv2
import numpy
import pprint
import random
import math
import time

cap = cv2.VideoCapture(0)

def create_and_train_model_from_dict(label_matrix):
    """ Create eigenface model from dict of labels and images """
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(label_matrix.values(), numpy.array(label_matrix.keys()))
    return model

def predict_image_from_model(model, image):
    """ Given an eigenface model, predict the label of an image"""
    return model.predict(image)

def read_csv(filename='faces.csv'):
    """ Read a csv file """
    csv = open(filename, 'r') # 'r' means read
    return csv

def prepare_training_data(file):
    """ prepare testing and training data from file"""
    lines = file.readlines()
    
    training_data = lines
    #training_data, testing_data = split_test_training_data(lines)
    
    return training_data
    #return training_data, testing_data

def create_label_matrix_dict(input_file):
    """ Create dict of label -> matricies from file """
    ### for every line, if key exists, insert into dict, else append
    label_dict = {}

    for line in input_file:
        ## split on the ';' in the csv separating filename;label
        filename, label = line.strip().split(';')

        ##update the current key if it exists, else append to it
        if label_dict.has_key(int(label)):
            current_files = label_dict.get(label)
            numpy.append(current_files,read_matrix_from_file(filename))
        else:
            label_dict[int(label)] = read_matrix_from_file(filename)

    return label_dict 


def read_matrix_from_file(filename):
    """ read in grayscale version of image from file """
    #return cv2.imread(filename, cv2.CV_IMREAD_GRAYSCALE)
    return cv2.imread(filename, 0)

if __name__ == "__main__":
    
    training_data = prepare_training_data(read_csv())
    data_dict = create_label_matrix_dict(training_data)
    #print data_dict.keys()
    model = create_and_train_model_from_dict(data_dict) 

    while(cap.isOpened()):
        ret, img = cap.read()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GREY)
        
        #for line in testing_data:
         #   filename, label =  line.strip().split(';')
        predicted_label = predict_image_from_model(model, grey)
        print predicted_label[0]
        
        
        
        if predicted_label[0] == 1:
            cv2.putText(img,'Hand detected', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
        else: 
            cv2.putText(img,'Put hand in here', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 1)
                
        cv2.imshow('Hand_Rec',img)
        
        #start = time.time()
        #while not (time.time() - start > 1):
        #    pass        
        time.sleep(0.1)
        if cv2.waitKey(10) == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
