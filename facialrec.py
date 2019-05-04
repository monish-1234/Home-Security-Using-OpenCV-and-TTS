# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:12:34 2019

@author: Monish
"""
import face_recognition
import cv2
import numpy as np 
import pyttsx3

engine = pyttsx3.init()

#REMOVE THE TWO LINES BELOW IF YOU'RE NOT RUNNING IN WINDOWS 10

en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
engine.setProperty('voice', en_voice_id)




video_capture = cv2.VideoCapture(0)

i1 = face_recognition.load_image_file("i1.jpg") # Image 1 
ie1 = face_recognition.face_encodings(i1)[0]

i2 = face_recognition.load_image_file("i2.jpg") # Image 2
ie2 = face_recognition.face_encodings(i2)[0]

i3 = face_recognition.load_image_file("i3.jpg") # Image 3 
ie3 = face_recognition.face_encodings(i3)[0]


known_face_encodings = [ie1,ie2,ie3]
known_face_names = ["Elon Musk","Bill Gates","Gal Gadot"] # Name of the person in that image 

# Facial Recognition Code Below
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
  
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
           
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                #Output Voice if a Recognised face appears
                engine.say(" Welcome Home "+name)

                engine.runAndWait()
            elif name=='Unknown':
                #Output voice if the face is not recognised
                engine.say("Intruder ALERT")
                engine.runAndWait()
            
             


            face_names.append(name)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to quit 
        break  
    process_this_frame = not process_this_frame

#Displaying the rectangle box with name
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#Press q to quit 
        break

video_capture.release()
cv2.destroyAllWindows()