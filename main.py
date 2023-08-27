from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from classifier import training

import tkinter as tk
from tkinter import *
import cv2
import os
from PIL import Image
import numpy as np
import facenet
import detect_face
import time
import pickle
import tensorflow.compat.v1 as tf



class MainProject:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x720+0+0")
        self.root.title("Autonomous Face Detection and Image Recognition Drone System")
        self.window = self.root

        self.Notification = tk.Label(self.root, text="All things good", bg="Green", fg="white", width=15, height=3)

        self.root.grid_rowconfigure(0, weight=1)  # Move this line to here
        self.root.grid_columnconfigure(0, weight=1)

        message = tk.Label(self.root, text="Autonomous Face Detection and Image Recognition Drone System", bg="cyan",
                           fg="black", width=50, height=3, font=('times', 30, 'italic bold '))
        message.place(x=63, y=20)

        lbl2 = tk.Label(self.root, text="Enter Name :", width=20, fg="black", height=2,
                        font=('times', 20, 'italic bold '))
        lbl2.place(x=370, y=300)

        self.txt2 = tk.Entry(self.root, width=20, fg="black")
        self.txt2.place(x=690, y=325)

        takeImg = tk.Button(self.root, text="Take Images", command=self.take_img, fg="black", bg="green", width=20,
                            height=3,
                            activebackground="Red", font=('times', 20, 'italic bold '), cursor="hand2")
        takeImg.place(x=100, y=500)

        trainImg = tk.Button(self.root, text="Train Images", command=self.trainimg, fg="black", bg="green", width=20,
                             height=3,
                             activebackground="Red", font=('times', 20, 'italic bold '))
        trainImg.place(x=475, y=500)

        recognizeImg = tk.Button(self.root, text="Face Recognition",command=self.face_recognition ,fg="black", bg="green",
                                 width=20, height=3,
                                 activebackground="Red", font=('times', 20, 'italic bold '))
        recognizeImg.place(x=850, y=500)

    def take_img(self):
        # Initialize the webcam
        cap = cv2.VideoCapture(0)  # 0 indicates the default webcam
        # Ask for the name to save the images
        name = self.txt2.get()
        path = 'train_img'
        print(path)
        directory = os.path.join(path, name)
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Number of images to capture
        num_images_to_capture = 50
        number_of_images = 0
        count = 0

        while number_of_images < num_images_to_capture:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break  # Break the loop if there's a problem capturing frames

            # Display the frame without bounding box
            cv2.imshow('Collecting The Dataset', frame)

            if count == 5:
                # Save the image
                image_filename = os.path.join(directory, f"{name}_{number_of_images}.jpg")
                cv2.imwrite(image_filename, frame)
                number_of_images += 1
                count = 0
                print(f"Image saved as {image_filename}")

            count += 1

            # Capture user input
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Dataset completed!...")

        res = "Images Saved Successfully : " + "\n" + " Name : " + name
        self.Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        self.Notification.place(x=290, y=400)

    def trainimg(self):

        from preprocess import preprocesses

        input_datadir = './train_img'
        output_datadir = './aligned_img'

        obj = preprocesses(input_datadir, output_datadir)
        nrof_images_total, nrof_successfully_aligned = obj.collect_data()

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

        datadir = './aligned_img'
        modeldir = './model/20180402-114759.pb'
        classifier_filename = './class/classifier.pkl'
        print("Training Start")
        obj = training(datadir, modeldir, classifier_filename)
        get_file = obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)

        res = "Model Trained Successfully"
        self.Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        self.Notification.place(x=290, y=400)



    def face_recognition(self):

        video = 0
        modeldir = './model/20180402-114759.pb'
        classifier_filename = './class/classifier.pkl'
        npy = './npy'
        train_img = "./train_img"
        imgBackground = cv2.imread("back.jpg")

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
                minsize = 30  # minimum size of face
                threshold = [0.7, 0.8, 0.8]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                batch_size = 100  # 1000
                image_size = 182
                input_image_size = 160
                HumanNames = os.listdir(train_img)
                HumanNames.sort()
                print('Loading Model...')
                facenet.load_model(modeldir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                classifier_filename_exp = os.path.expanduser(classifier_filename)

                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile, encoding='latin1')

                video_capture = cv2.VideoCapture(video)
                print('Start Recognition')
                while True:
                    ret, frame = video_capture.read()
                    # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                    timer = time.time()

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)

                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    faceNum = bounding_boxes.shape[0]

                    if faceNum > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]
                        cropped = []
                        scaled = []
                        scaled_reshape = []

                        for i in range(faceNum):
                            emb_array = np.zeros((1, embedding_size))
                            xmin = int(det[i][0])
                            ymin = int(det[i][1])
                            xmax = int(det[i][2])
                            ymax = int(det[i][3])

                            try:
                                # inner exception
                                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[ymin:ymax, xmin:xmax, :])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]

                                if best_class_probabilities > 0.87:
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # boxing face
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names = HumanNames[best_class_indices[0]]
                                            accuracy = best_class_probabilities[0]  # Get the accuracy value
                                            print("Predictions : [ name: {}, accuracy: {:.3f} ]".format(result_names,
                                                                                                        accuracy))
                                            cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin - 2), (0, 255, 255), -1)
                                            cv2.putText(frame, f"{result_names} ({accuracy:.3f})", (xmin, ymin - 5),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1,
                                                        lineType=1)
                                else:
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                    cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin - 2), (0, 255, 255), -1)
                                    cv2.putText(frame, "Unknown Face", (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)
                            except:
                                print("error")

                    endtimer = time.time()
                    fps = 1 / (endtimer - timer)
                    cv2.rectangle(frame, (15, 30), (135, 60), (0, 255, 255), -1)
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    imgBackground[162:162 + 480, 55:55 + 640] = frame
                    cv2.imshow("Autonomous Face Detection and Image Recognition Drone System", imgBackground)
                    key = cv2.waitKey(1)

                    if key == 113:  # "q"
                        break

                video_capture.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = MainProject(root)
    root.mainloop()
