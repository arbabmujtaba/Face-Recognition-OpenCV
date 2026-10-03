import cv2
import os
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("face-attendence-3e008-firebase-adminsdk-fbsvc-b3f5e0c51c.json")
firebase_admin.initialize_app(cred,{
    "databaseURL":"https://face-attendence-3e008-default-rtdb.firebaseio.com/",
    "storageBucket":"face-attendence-3e008.appspot.com"

})
# Download models if not found
yunetpath = "face_detection_yunet_2023mar.onnx"
sfacepath = "face_recognition_sface_2021dec.onnx"

detector = cv2.FaceDetectorYN.create(
    yunetpath, "", (640, 480), 0.5, 0.3, 5000
)

recognizer = cv2.FaceRecognizerSF.create(sfacepath, "")

# importing the student images
folderpath = 'images'   # making the path changable
pathlist = os.listdir(folderpath)  # listing the directories

Imglist = []
studentids = []

for path in pathlist:  # yeahn p list se iterate hota hai
    imagepath = os.path.join(folderpath, path)
    img = cv2.imread(imagepath)


    
    fileName = os.path.join(folderpath, path)
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)




    if img is None:
        print("Could not read image:", path)
        continue

    Imglist.append(img)
    studentids.append(os.path.splitext(path)[0])

print(len(Imglist))      #
print(studentids)

# giving the encoding function some images and generating them
def findEncodings(imageList):
    Encodelist = []

    for i, img in enumerate(imageList):
        image_name = studentids[i]
        print("Processing image:", image_name)

        height, width = img.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(img)

        if faces is None or len(faces) == 0:
            print("No face detected in:", image_name)
            raise ValueError("No face detected in image: " + image_name)

        # Select the largest face
        face = max(faces, key=lambda f: f[2] * f[3])

        print("Faces detected in", image_name + ":", len(faces))
        print("Selected face for encoding:", image_name)

        faceAligned = recognizer.alignCrop(img, face)
        Encode = recognizer.feature(faceAligned)
        Encodelist.append(Encode.flatten())

        print("Encoding complete for:", image_name)

    return Encodelist

print("Encoding_Started.........")
EncodeListKnown = findEncodings(Imglist)
EncodeListKnownWithIds = [EncodeListKnown, studentids]
print(EncodeListKnown)
print("Encoding Complete")

file = open("EncodeFile.p", "wb")
pickle.dump(EncodeListKnownWithIds, file)
file.close()
print("File Saved My Darling")
