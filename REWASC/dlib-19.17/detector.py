import dlib
import os
import glob
import cv2

detectorLatas = dlib.simple_object_detector("resources/latas.svm")
detectorGarrafas = dlib.simple_object_detector("resources/garrafas.svm")

detectors = [detectorLatas, detectorGarrafas]


for imagem in glob.glob(os.path.join("Imagens_Validacao", "*")):

    image = cv2.imread(imagem)
    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1, adjust_threshold=0.0)
    latas_detectadas = detectorLatas(image, 1)
    garrafas_detectadas = detectorGarrafas(image, 1)
    cont = 0

    for i in range(len(boxes)):
        cont += 1
        if confidences[i] > 0.3:
            if(detector_idxs[i] == 0):
                print("Tipo LATA com confiança de {}.".format(confidences[i]))
            elif(detector_idxs[i] == 1):
                print("Tipo GARAFFA com confiança de {}.".format(confidences[i]))



        cv2.imshow("Multiclasses", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()


