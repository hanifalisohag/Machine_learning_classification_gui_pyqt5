import sys
# Import required packages:
import tensorflow as tf
import numpy as np
import os

import cv2
import PyQt5.QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image

class Life2Coding(QMainWindow):
    def __init__(self):
        super(Life2Coding,self).__init__()
        loadUi('life2coding_final.ui',self)
        self.image=None

        self.dirName = None
        self.imgFileName = None
        self.netResnet = load_model('./models/model-resnet50-final.h5')
        # self.firmware_dir = None

        self.last_firmware_directory = None
        self.cls_list = ['hipster', 'not_hipster']

        self.btn_load.clicked.connect(self.loadClicked)
        self.btn_predict.clicked.connect(self.crop_predict)

        # self.btn_predict.clicked.connect(self.crop_predict_HAAR)
    @pyqtSlot()
    def loadClicked(self):
        # fname,filter =QFileDialog.getOpenFileName(self,'Open File','C:\\',"JPG Files (*.jpg);;PNG files (*.png)")

        ret=self.loadImageFile()

        print("file name",ret)
        if 'p#' in ret:
            self.lbl_result.setText("Please Load File Properly")
            print('Invalid Image')
        else:
            self.loadImage(ret)

    def loadImageFile(self):

        filter = "Image files (*.jpg *.png *.jpeg)"
        firmware_dir = None
        if self.last_firmware_directory:
            firmware_dir = self.last_firmware_directory

        p = QFileDialog.getOpenFileName(parent=self, caption="Select Image File",
                                        directory=firmware_dir, filter=filter)
        path = p[0]
        if path:
            # self.firmwarePathEdit.setText(path)
            self.last_firmware_directory = "/".join(path.split("/")[0:-1])

        if p[0]:
            self.imgFileName = p[0]
            print(p[0])
            return self.imgFileName
        else:
            print('Invalid Image File')
            return "p#"



    def loadImage(self,fname):
        self.image=cv2.imread(fname,cv2.IMREAD_COLOR)
        self.lbl_result.setText("")
        self.displayImage(self.image)


    def displayImage(self, img):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # index[0]=rows,index[1]=cols,index[2]=channels
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR > RGB
        outImg = outImg.rgbSwapped()

        self.pixmap = QPixmap.fromImage(outImg)

        self.lbl_display.setPixmap(self.pixmap.scaled(self.lbl_display.size(), PyQt5.QtCore.Qt.KeepAspectRatio,
                                                      PyQt5.QtCore.Qt.SmoothTransformation))
        # self.lbl_display.setPixmap(QPixmap.fromImage(img))
        self.lbl_display.setAlignment(PyQt5.QtCore.Qt.AlignHCenter | PyQt5.QtCore.Qt.AlignVCenter)



    def addFacePadding(self,cvRect, padding):
        cvRect[0] = cvRect[0] - padding  # left x
        cvRect[1] = cvRect[1] - padding  # left y

        cvRect[2] = cvRect[2] + padding  # bottom x
        cvRect[3] = cvRect[3] + padding  # bottom y

        return cvRect

    def predict_image(self, img, debug=True):

        try:

            # Path of  training images
            train_path = r'./data/train'
            if not os.path.exists(train_path):
                print("No such directory")
                raise Exception

            image_size = 128
            num_channels = 3
            images = []

            # Resizing the image to our desired size and preprocessing will be done exactly as done during training
            image1 = cv2.resize(img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            images.append(image1)
            images = np.array(images, dtype=np.uint8)
            images = images.astype('float32')
            images = np.multiply(images, 1.0 / 255.0)

            # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
            x_batch = images.reshape(1, image_size, image_size, num_channels)

            # Let us restore the saved model
            sess = tf.Session()
            # Step-1: Recreate the network graph. At this step only graph is created.
            saver = tf.train.import_meta_graph('models/trained_model.meta')
            # Step-2: Now let's load the weights saved using the restore method.
            saver.restore(sess, tf.train.latest_checkpoint('./models/'))

            # Accessing the default graph which we have restored
            graph = tf.get_default_graph()

            # Now, let's get hold of the op that we can be processed to get the output.
            # In the original network y_pred is the tensor that is the prediction of the network
            y_pred = graph.get_tensor_by_name("y_pred:0")

            ## Let's feed the images to the input placeholders
            x = graph.get_tensor_by_name("x:0")
            y_true = graph.get_tensor_by_name("y_true:0")
            y_test_images = np.zeros((1, len(os.listdir(train_path))))

            # Creating the feed_dict that is required to be fed to calculate y_pred
            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)
            # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
            if debug:
                print(result)

            # Convert np.array to list
            a = result[0].tolist()
            r = 0

            # Finding the maximum of all outputs
            max1 = max(a)
            index1 = a.index(max1)
            predicted_class = None

            # Walk through directory to find the label of the predicted output
            count = 0
            for root, dirs, files in os.walk(train_path):
                for name in dirs:
                    if count == index1:
                        predicted_class = name
                    count += 1

            # If the maximum confidence output is largest of all by a big margin then
            # print the class or else print a warning
            for i in a:
                if i != max1:
                    if max1 - i < i:
                        r = 1
            if r == 0:
                if debug:
                    print("Predicted:", predicted_class)
            else:
                if debug:
                    print("Could not classify with definite confidence")
                    print("Maybe:", predicted_class)

            return predicted_class

        except Exception as e:
            print("Exception:", e)


    def crop_predict(self, padding=5):
        count=0
        detected=0
        # padding = 5
        # Load pre-trained model:
        net = cv2.dnn.readNetFromCaffe("./models/deploy.prototxt",
                                       "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

        # Load images and get the list of images:
        # image = cv2.imread("face.jpg")

        # frameHeight = image.shape[0]
        mainImage = self.image.copy()
        frameWidth = mainImage.shape[1]
        padPercent = int(frameWidth * padding / 100)

        (h, w) = mainImage.shape[:2]
        # Call cv2.dnn.blobFromImages():
        blob_images = cv2.dnn.blobFromImage(cv2.resize(mainImage, (300, 300)), 1.0, (250, 250), [104., 117., 123.],False,False)

        # Set the blob as input and obtain the detections:
        net.setInput(blob_images)
        detections = net.forward()

        # print("go",detections.shape[2])
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak predictions:
            if confidence > 0.7:
                count+=1
                # Get the size of the current image:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")


                cvRect = [int(startX), int(startY),
                          int(endX), int(endY)]

                cvRect = self.addFacePadding(cvRect, padPercent)




                if cvRect[1] < 0:
                    cvRect[1] = 0
                elif cvRect[2] < 0:
                    cvRect[2] = 0
                elif cvRect[3] < 0:
                    cvRect[3] = 0
                elif cvRect[0] < 0:
                    cvRect[0] = 0

                y = endY + 15 if endY - 10 > 10 else endY + 15

                print('pos=', h, w)
                print('rec', cvRect)

                if cvRect[2] > w or cvRect[3] > h:
                    print('wrong')
                else:

                    roi = mainImage[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]

                    roi = cv2.resize(roi, (224, 224))

                    if roi is None:
                        continue
                    x = image.img_to_array(roi)
                    x = preprocess_input(x)
                    x = np.expand_dims(x, axis=0)
                    pred = self.netResnet.predict(x)[0]
                    top_inds = pred.argsort()[::-1][:5]
                    # print(f)
                    for i in top_inds:
                        print('    {:.3f}  {}'.format(pred[i], self.cls_list[i]))

                    # print("predict: ",self.cls_list[np.argmax(pred, axis=0)])

                    ind=np.argmax(pred, axis=0)

                    result=self.cls_list[ind]

                    text = "{:.2f}%".format(pred[ind] * 100)
                    text2 = "{}".format(result)

                    # cv2.rectangle(mainImage, (startX, startY), (endX, endY),
                    #               (0, 0, 255), 2)

                    cv2.rectangle(mainImage, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]),
                                  (0, 255, 255), 2)

                    cv2.putText(mainImage, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    #
                    cv2.putText(mainImage, text2, (startX, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(count)
        if count>0:
            text1 = "Predicted: {}".format(result)
            self.lbl_result.setText(text1)
            # self.lbl_result.setText(result)
            self.displayImage(mainImage)
        else:
            self.lbl_result.setText("Please Use a Full Front Face")



if __name__=='__main__':
    app=QApplication(sys.argv)
    window=Life2Coding()
    window.setWindowTitle('Hipster or Not Detection by Md Hanif Ali Sohag (Contact: hanifalisohag@gmail.com)')
    window.show()
    sys.exit(app.exec_())