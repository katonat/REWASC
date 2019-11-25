import dlib


options = dlib.simple_object_detector_training_options()
# options.add_left_right_image_flips = True
options.C = 5
# options.epsilon = 0.1
options.be_verbose = True


dlib.train_simple_object_detector("dlib-19.17/resources/latas.xml", "dlib-19.17/resources/latas.svm", options)


#
# print("")  # Print blank line
#
# # print(the precision, recall, and then)
# print("Training accuracy: {}".format(
#     dlib.test_simple_object_detector("dlib-19.17/resources/latas1.xml", "dlib-19.17/resources/latas800C100Y01.svm")))
#
# print("Testing accuracy: {}".format(
#     dlib.test_simple_object_detector("dlib-19.17/resources/latas1.xml", "dlib-19.17/resources/latas800C100Y01.svm")))