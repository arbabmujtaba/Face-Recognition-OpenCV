
import cv2
import os
# import EncodeGenerator  # CHANGED: Not needed here
import pickle
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

background = cv2.imread('resources/background.png')

## importing the modeimages into a list
foldermodepath = 'resources/Modes'   # making the path changable
modepathlist = os.listdir(foldermodepath)  # listing the directories
imagemodelist = []

for path in modepathlist:
    imagemodelist.append(cv2.imread(os.path.join(foldermodepath, path)))

print(modepathlist)
print(len(imagemodelist))

# load the ecoding file brother
print("Loading Encode File....")

file = open("EncodeFile.p", "rb")
EncodeListKnowWithIds = pickle.load(file)
file.close()

EncodeListKnown, studentids = EncodeListKnowWithIds
print("Encode File Loaded")
# print(studentids)

# CHANGED: Run face recognition only every 10 frames
frame_count = 0

while True:
    sucess, img = cap.read()

    # CHANGED: Check if camera successfully reads a frame
    if not sucess:
        print("Camera frame not received")
        break

    frame_count += 1

    # Shrinking The Size Kunki Zayda computation lagti hai
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # CHANGED: Face recognition runs every 10 frames
    if frame_count % 10 == 0:
        # CHANGED: Use RGB image and HOG detector
        FaceCurFrame = face_recognition.face_locations(imgs, model="hog")
        EncodeCurFrame = face_recognition.face_encodings(imgs, FaceCurFrame)

        # Now comparing with the General Encodings we did earlier
        for encodeface, faceloc in zip(EncodeCurFrame, FaceCurFrame):
            Matches = face_recognition.compare_faces(EncodeListKnown, encodeface)
            FaceDis = face_recognition.face_distance(EncodeListKnown, encodeface)

            # print("matches", Matches)
            # print("faceDistance", FaceDis)

            # CHANGED: Avoid error if no encodings are available
            if len(FaceDis) > 0:
                MatchIndex = np.argmin(FaceDis)
                print("Match_Index", MatchIndex)
            if Matches[MatchIndex]:
                print("Known Face Detected")
                print(studentids[MatchIndex])
    # Webcam
    img = cv2.resize(img, (805, 605))
    background[290:290+605, 44:44+805] = img

    # Mode image - 10% bigger
    modeImg = cv2.resize(imagemodelist[3], (568, 990))
    background[60:60+990, 880:880+568] = modeImg
    cv2.imshow("Face Attendance", background)


    # CHANGED: Press Q to exit and release the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CHANGED: Release camera and close windows
cap.release()
cv2.destroyAllWindows()