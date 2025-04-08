#Our  model is in the format of ".h5" now lets load the model. Keep the .h5 file  in the same directory . 


import tensorflow as tf 
import  matplotlib.pyplot as plt 
import matplotlib.image as mpimg


model = tf.keras.models.load_model("Model_2.h5")

#Our  default class_names is as ['Normal' , 'Pneumonia']

class_names=['Normal','Pneumonia']

#Creating a helper functions using which we can display the prediction
def show_result(model,filepath,class_names):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_image(image)
    if image.shape[:,-1]==3: 
        image = tf.image.rgb_to_gray(image)
    image = tf.image.resize(image,size=[224,224])#because my  model "EfficinetB0" excepts image size qs(224,224)
    pred_probs = model.predict(tf.expand_dims(image,axis=0))
    if pred_probs > 0.8:
        pred_class =class_names[int( tf.argmax(pred_probs,axis=1))]
        plt.imshow(image/255)
        plt.title("f{pred_class}")
        plt.show()


show_result(model = model,filepath = "../...",class_names=class_names)#filepath must be the path where our image is saved

    
#This python file is able to predict when a image is given to it , and also able to load the model that is being downloaded in our device.






