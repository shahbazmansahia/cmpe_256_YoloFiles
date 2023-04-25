import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


import os;
import tqdm;
import pandas as pd;

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    ### custom code starts
    is_dir = False;
    print ("Path: ", images[0], "\n");
    enumerator = enumerate(images, 1);

    count_list = []
    f_name_list = []
    score_list = []
    class_list = []
    class_label_list = []

    if os.path.isdir(images[0]):
        is_dir = True;
        # counts for files and then redefines enumerator to go over multiple files in a directory!
        file_names = os.listdir(images[0]);
        items = [images[0] + s for s in file_names]
        print("Items: \n", items, "\n");
        print("Item #: \n", len(items), "\n");
        enumerator = enumerate (items, 0);
        #enumerator = enumerate (tqdm(items));


    print ("is_dir set to: ", is_dir, "\n");


    ### custom code ends

    # load model
    if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each

    for count, image_path in enumerator:

        ### custom code starts
        print ("count: ", count, "\n", "image_path: ", image_path, "\n");
        ### custom code ends

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if FLAGS.framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        ### cusom code starts here

        #print ("boxes type: ", type(boxes.numpy()), "\n")
        #print ("boxes: ", boxes.numpy(), "\n")

        #print ("scores type: ", type(scores.numpy()), "\n")
        #print ("scores: ", scores.numpy(), "\n")
        #for i in scores.numpy():
        #    print (print ("scores: ", len(i), "\n"))

        #print ("classes type: ", type(classes.numpy()), "\n")
        #print ("classes: ", classes.numpy(), "\n")
        #for i in classes.numpy():
        #    print (print ("classes: ", len(i), "\n"))


        #print ("valid_detections type: ", type(valid_detections.numpy()), "\n")
        #print ("valid_detections: ", valid_detections.numpy(), "\n")
        ### cusom code end here

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

        image = Image.fromarray(image.astype(np.uint8))
        if not FLAGS.dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        #cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

        ### cusom code starts here
        # to rename file the same as object
        cv2.imwrite(FLAGS.output + file_names[count] + '.png', image)

        print ("Item processed: ", count,"\nItem Path: ", image_path , "\n")

        #print ("class_names: ", class_names, "\n")
        #for i in scores.numpy():
        #    for j in range(len(i)):
        #        if (i[j] > 0):
        #            print (class_names[j]);

        #for i in scores.
        # logging values
        try:

            #df = pd.DataFrame()
            with open (FLAGS.output + "detection.txt", 'a') as f:
                #f.write (file_names[count] + "," + str(scores.numpy()[:]) + "," + classes.numpy()[:])

                #f.write (file_names[count] + ",")

                scores_str = "[ ";
                # for writing scores to file
                for i in scores.numpy():
                    #f.write(str(i) + " ")
                    scores_str = scores_str + str(i)[:-1] + ",";
                    print ("5th I in scores:", str(i[5]))
                    print ("6th I in scores:", str(i[6]))
                    ### test
                    score_list.append(i)
                    #print (scores_str)

                #f.write ("],[")
                scores_str = scores_str[:-1] + "]"

                classes_str = "[ "
                # for writing classes to file
                for i in classes.numpy():
                    #f.write(str(i) + " ")
                    classes_str = classes_str + str(i)[:-1] + ",";
                    class_list.append(i)

                #f.write ("],[")
                classes_str = classes_str[:-1] + "]"

                class_n_list  = "["
                # for getting list of detected objects from classes
                print ("class_names: ", class_names, "\n")
                for i in scores.numpy():
                    for j in range(len(i)):
                        if (i[j] > 0):
                            #f.write(class_names[j] + " ")
                            class_n_list = class_n_list + class_names[j] + " "
                            class_label_list.append(class_names[j])

                #f.write ("]\n")
                if len(class_n_list) > 1:
                    class_n_list = class_n_list[:-1] + "]"

                else:
                    class_n_list = class_n_list + "]"

                count_list.append(count)
                f_name_list.append(file_names[count])
                #line = [str(count), file_names[count], scores_str, classes_str, class_n_list]

                f.write (str(count) + "|" + file_names[count] + "|" + scores_str + "|" + classes_str + "|" + class_n_list + "\n")
                #f.write(line)


                print (str(count) + "\n File Name: " + file_names[count] + "\nScores: " + scores_str + "\nClasses: " + classes_str + "\nClass List: " + class_n_list + "\n")



                print ("Creating text file for logging values...")

                ### test test block starts
                try:

                    #df = pd.DataFrame()
                    with open (FLAGS.output + "filenames.txt", 'a') as f:
                        counter = 0
                        for i in f_name_list:
                            f.write(str(i) + "\n")
                except:
                    print ("filenames error!")
                    pass;

                try:

                    #df = pd.DataFrame()
                    with open (FLAGS.output + "scores.txt", 'a') as f:
                        counter = 0
                        for i in score_list:
                            f.write(str(i) + "\n")
                except:
                    print ("scores error!")
                    pass;

                try:

                    #df = pd.DataFrame()
                    with open (FLAGS.output + "classes.txt", 'a') as f:
                        counter = 0
                        for i in class_list:
                            f.write(str(i) + "\n")
                except:
                    print ("class error!")
                    pass;
                try:

                    #df = pd.DataFrame()
                    with open (FLAGS.output + "class_names.txt", 'a') as f:
                        counter = 0
                        for i in class_label_list:
                            f.write(str(i) + "\n")
                except:
                    print ("class_label error!")
                    pass;
                ### test test block ends

            #df = pd.DataFrame(count)
            #df['file_names'] = f_name_list


        except FileNotFoundError:
            print ("File Not Found!")
            #counter = counter + 1


        ### custom code ends here

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
