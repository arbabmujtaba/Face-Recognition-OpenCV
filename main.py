import cv2
import os
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("face-attendence-3e008-firebase-adminsdk-fbsvc-b3f5e0c51c.json")
firebase_admin.initialize_app(cred,{
    "databaseURL":"https://face-attendence-3e008-default-rtdb.firebaseio.com/",
    "storageBucket":"face-attendence-3e008.appspot.com"

})
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

background = cv2.imread('resources/background.png')

## importing the modeimages into a list
foldermodepath = 'resources/Modes'   # making the path changable
modepathlist = sorted(os.listdir(foldermodepath))  # listing the directories
imagemodelist = []

for path in modepathlist:
    imagemodelist.append(cv2.imread(os.path.join(foldermodepath, path)))

# print(modepathlist)
# print(imagemodelist[0])
# using the models for the detection
yunetpath = "face_detection_yunet_2023mar.onnx"
sfacepath = "face_recognition_sface_2021dec.onnx"
# Load YuNet and SFace models
print("Loading YuNet Model....")
detector = cv2.FaceDetectorYN.create(
    yunetpath, "", (640, 480), 0.5, 0.3, 5000
)
print("YuNet Model Loaded")

print("Loading SFace Model....")
recognizer = cv2.FaceRecognizerSF.create(sfacepath, "")
print("SFace Model Loaded")

# load the ecoding file brother
print("Loading Encode File....")

file = open("EncodeFile.p", "rb")
EncodeListKnowWithIds = pickle.load(file)
file.close()

EncodeListKnown, studentids = EncodeListKnowWithIds
print("Encode File Loaded")

# Run face recognition only every 10 frames
frame_count = 0
bbox = []


ModeType = 0
totalAttendance = None
studentinfo = None

counter = 0
id = -1
while True:
    sucess, img = cap.read()

    # Check if camera successfully reads a frame
    if not sucess:
        print("Camera frame not received")
        break

    frame_count += 1

    # Face recognition runs every 10 frames
    if frame_count % 10 == 0:
        height, width = img.shape[:2]
        detector.setInputSize((width, height))
        _, FaceCurFrame = detector.detect(img)

        bbox = []
        if FaceCurFrame is not None:
            # print("Faces detected:", len(FaceCurFrame))
            pass

            for face in FaceCurFrame:
                x, y, w, h = face[:4].astype(int)
                bbox.append((x, y, w, h))

                faceAligned = recognizer.alignCrop(img, face)
                EncodeCurFrame = recognizer.feature(faceAligned).flatten()

                if len(EncodeListKnown) > 0:
                    EncodeCurFrame = np.asarray(EncodeCurFrame, dtype=np.float32).reshape(1, -1)
                    FaceDis = [
                        recognizer.match(
                            EncodeCurFrame,
                            np.asarray(knownEncode, dtype=np.float32).reshape(1, -1),
                            cv2.FaceRecognizerSF_FR_COSINE
                        )
                        for knownEncode in EncodeListKnown
                    ]

                    MatchIndex = int(np.argmax(FaceDis))
                    FaceScore = FaceDis[MatchIndex]
                    id = studentids[MatchIndex]

                    if counter == 0:
                        counter += 1
            if counter == 1:
                print("Match_Index", MatchIndex)
                print("Face_Distance", FaceScore)

                if FaceScore >= 0.450:
                    print("Known Face Detected")
                    print("Student ID:", studentids[MatchIndex])
                    ModeType = 1
                    try:
                        studentinfo = db.reference(f'Students/{id}').get()
                        print(studentinfo)
                        # update data of attendance 
                        # updating the time and attendance
                        DateTimeObject = datetime.strptime(studentinfo['last_attendance_time'],"%Y-%m-%d %H:%M:%S")
                        secondsElapsed =(datetime.now()-DateTimeObject).total_seconds()
                        datetime__ = datetime.now()
                        print(secondsElapsed)


                        if secondsElapsed > 15:

                        # updating the Database
                            ref = db.reference(f'Students/{id}')
                            studentinfo['total_attendance'] += 1
                            ref.update({'total_attendance': studentinfo['total_attendance']})
                            ref.update({'last_attendance_time': datetime__.strftime("%Y-%m-%d %H:%M:%S")})
                            totalAttendance = str(studentinfo['total_attendance'])
                            frame_count = 0
                        else: 
                            ModeType = 3
                        # if counter>=0:
                    except Exception as e:
                        print("Firebase error:", e)
                    # Data fetched once; don't query again until the face leaves
                    counter = 2
                else:
                    print("Unknown Face Detected")
        else:
            counter = 0
            ModeType = 0
            totalAttendance = None
            studentinfo = None

    for x, y, w, h in bbox:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    # Webcam
    img = cv2.resize(img, (805, 605))
    background[290:290+605, 44:44+805] = img

    # Mode image - 10% bigger
    modeImg = cv2.resize(imagemodelist[ModeType], (568, 990))
    background[60:60+990, 880:880+568] = modeImg

    if 60<frame_count<90:
        ModeType=2
        modeImg = cv2.resize(imagemodelist[ModeType], (568, 990))
        background[60:60+990, 880:880+568] = modeImg



    # here we are 0verlaying the details 
    if totalAttendance is not None and counter <=10 and 0<frame_count<60:
        cv2.putText(background, str(studentinfo['total_attendance']),(955,190), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['name']),(965,700), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['major']),(1147,852), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(background, str(id),(1147,765), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['standing']),(1020,984), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['year']),(1172,984), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['starting_year']),(1300,976), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0,0,0), 1)
    cv2.imshow("Face Attendance", background)

    # Press Q to exit and release the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
