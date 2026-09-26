import cv2
import os
import face_recognition
import pickle
# importing the student images



folderpath = 'images'   # making the path changable 

pathlist = os.listdir(folderpath)  #listing the directories

Imglist = []
studentids = []
for path in pathlist:                       #yeahn p list se iterate hota hai
    Imglist.append(cv2.imread(os.path.join(folderpath,path)))    # what exacly did he do here
    studentids.append(os.path.splitext(path)[0])

print(len(Imglist))
print(studentids)


# giving the encoding function some images and generating them
def findEncodings(imageList):
    Encodelist = []
    for img in imageList:

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        Encode = face_recognition.face_encodings(img)[0]
        Encodelist.append(Encode)

    return Encodelist
print("Encoding_Started.........")
EncodeListKnown = findEncodings(Imglist)
EncodeListKnownWithIds = [EncodeListKnown,studentids]
print(EncodeListKnown)
print("Encoding Complete")



file = open("EncodeFile.p","wb")
pickle.dump(EncodeListKnownWithIds,file)
file.close()
print("File Saved My Darling")