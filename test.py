
import cv2
import os
import pickle
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

background = cv2.imread('resources/background.png')

# importing the modeimages into a list
foldermodepath = 'resources/Modes'
modepathlist = os.listdir(foldermodepath)
imagemodelist = []

for path in modepathlist:
    imagemodelist.append(cv2.imread(os.path.join(foldermodepath, path)))

print(modepathlist)
print(len(imagemodelist))

# load the encoding file
print("Loading Encode File....")

with open("EncodeFile.p", "rb") as file:
    EncodeListKnowWithIds = pickle.load(file)

EncodeListKnown, studentids = EncodeListKnowWithIds
print("Encode File Loaded")

frame_count = 0
EncodeCurFrame = []
FaceCurFrame = []

try:
    while True:
        sucess, img = cap.read()

        if not sucess:
            print("Camera frame not received")
            break

        frame_count += 1

        # Shrinking the size to reduce computation
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgs = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Run face recognition every 10 frames
        if frame_count % 10 == 0:
            FaceCurFrame = face_recognition.face_locations(
                imgs, model="hog"
            )
            EncodeCurFrame = face_recognition.face_encodings(
                imgs, FaceCurFrame
            )

            for encodeface, faceloc in zip(EncodeCurFrame, FaceCurFrame):
                Matches = face_recognition.compare_faces(
                    EncodeListKnown, encodeface
                )
                FaceDis = face_recognition.face_distance(
                    EncodeListKnown, encodeface
                )

                print("matches", Matches)
                print("faceDistance", FaceDis)

                if len(FaceDis) > 0:
                    MatchIndex = np.argmin(FaceDis)
                    print("Match_Index", MatchIndex)

        # Webcam
        img = cv2.resize(img, (805, 605))
        background[290:290+605, 44:44+805] = img

        # Mode image
        modeImg = cv2.resize(imagemodelist[3], (568, 990))
        background[60:60+990, 880:880+568] = modeImg

        cv2.imshow("Face Attendance", background)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()