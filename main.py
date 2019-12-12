# Import required modules
import cv2 as cv
import numpy as np
from face_detection import detect_face
from text_detection import text_detector
from rotation import rotate_image, center_card
import matplotlib.pyplot as plt

image = cv.imread('./input/flipped card.jpg')

def detect_number(frame):

    # aks kart meli center shode amade mishe miad ( ba ROI anjam shode)
    # emkane in hast ke aks 90 daraje rotate shode bashe ya hata baraks bashe
    centered_card = center_card(frame);

    if type(centered_card) == type(None):
        centered_card = frame
    face_detected_image = detect_face(centered_card)

    i = 0;
    rotated_image = centered_card
    # inja face detection mizanim ke bebinim aksemoon saf shode ya na. detect kar yani safe
    # age detect nakard yani hanooz kaje pas 90 daraje hey rotate mikonim ta detect kone
    # bishtar az  4 bar am sense nemikone ke rotate konim chon bar migarde sare jash
    while type(face_detected_image) == type(None):
        rotated_image = rotate_image(rotated_image, 90)
        face_detected_image = detect_face(rotated_image)
        if i > 4:
            break;
        i += 1

    # age too 4 bar nashod neshoon mide dorost center nashode va khata mide
    if (type(face_detected_image) != type(None)):
        result = text_detector(face_detected_image);
    else:
        print('could not find ID number. sorry')
    return result

code_meli = detect_number(image)
plt.imshow(code_meli)
plt.show()