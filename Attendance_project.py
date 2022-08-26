import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Training_images'
images = []                 # this will contain images name with extension
personNames = []            # this will contain images name without extension
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    personNames.append(os.path.splitext(cl)[0])    # splitting image file name based on text
print(personNames)

# face_encodings() is based on dlib and this Dlib encode the image into 128 unique features means it searches 120 unique
# point from the image
def findEncodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # cv2 takes image in color format BGR we have to convert it into RGB
        encode = face_recognition.face_encodings(img)[0]   # finding 128 unique features using "HOG transformation algorithm"
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete!!!!!!!')

names = personNames
deadline = "10:42:00"
def attendance(name):
    if name == "Unknown":
        return
    now = datetime.now()
    timeString = now.strftime('%H:%M:%S')
    if timeString > deadline:
        print("Deadline Crosses!!!!!!!!!!!!!")
        myList = os.listdir()
        dateString = now.strftime('%d-%m-%Y')
        path = "Attend_Register_" + dateString + ".csv"
        if path in myList:
            myDataList = ""
            with open(path, 'r') as f:
                myDataList = f.readlines()
            with open(path,"w") as f:
                f.write(myDataList[0])
                f.write(myDataList[1])
                for i in range(2,len(myDataList)):
                    if "Present" in myDataList[i]:
                        s = myDataList[i].replace("\n", "")
                        f.write(f"{s}")
                    elif "Absent" in myDataList[i]:
                        s = myDataList[i].replace("\n", "")
                        f.write(f"{s}")
                    else:
                        s=myDataList[i].replace("\n","")+",Absent"
                        f.write(f"{s}")
                    if i<len(myDataList)-1:
                        f.write("\n")

    else:
        myList = os.listdir()
        dateString = now.strftime('%d-%m-%Y')
        path ="Attend_Register_"+dateString+".csv"

        if path in myList:
            myDataList =""
            with open(path, 'r') as f:
                myDataList = f.readlines()
            nameInFile = []
            for i in range(2,len(myDataList)):
                s=myDataList[i].replace("\n","")
                nameInFile.append(s)

            with open(path, "a") as f:
                for nm in names:
                    s1=nm+",Present"
                    s2=nm+",Absent"
                    if nm in nameInFile:
                        continue
                    elif s1 in nameInFile:
                        continue
                    elif s2 in nameInFile:
                        continue
                    else:
                        f.write(f"\n{nm}")

            with open(path, 'r') as f:
                myDataList = f.readlines()

            if (name+"\n") in myDataList[:]:
                myDataList[myDataList.index(name+"\n")] = name+",Present\n"
            elif name in myDataList[2:]:
                myDataList[myDataList.index(name)] = name+",Present"
            with open(path, 'w') as f:
                for nm in myDataList:
                    f.write(nm)

        else:
            with open(path, "a") as f:
                f.write(f"Date:-,{dateString}\n")
                f.write("Name,Attendance\n")
                for i in range(len(names)-1):
                    f.write(f"{names[i]}\n")
                f.write(f"{names[-1]}")
            attendance(name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()                                  # ret is a boolean variable that returns true if the frame is available. frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)      # resize the frame means we converted into one-forth of actual size of the screen
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)           # Camera takes image in color format BGR we have to convert it into RGB

    facesCurFrame = face_recognition.face_locations(faces)    # it will search only faces
    encodesCurFrame = face_recognition.face_encodings(faces, facesCurFrame)    # enconding faces captured by camera

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)       # comapre training faces to testing faces
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)       # facedistance is less means faces matches.
        # print(faceDis)
        matchIndex = np.argmin(faceDis)                                             # find the index of min face-distance

        if matches[matchIndex]:                                                     # if faces matches with training faces
            name = personNames[matchIndex]
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4                     # multiplying with 4 because we have reduces the frame width to one-forth
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 36), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            attendance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # multiplying with 4 because we have reduces the frame width to one-forth
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 40), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



    cv2.imshow('Webcam', frame)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()