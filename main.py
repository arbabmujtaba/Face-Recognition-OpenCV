import cv2
import os
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
background = cv2.imread('resources/background.png')
                   ## importing the modeimages into a list
foldermodepath = 'resources/Modes'   # making the path changable 

modepathlist = os.listdir(foldermodepath)  #listing the directories

imagemodelist = []


for path in modepathlist:                       #yeahn p list se iterate hota hai
    imagemodelist.append(cv2.imread(os.path.join(foldermodepath,path)))    # what exacly did he do here

print(modepathlist)
print(len(imagemodelist))
while True:

    sucess, img = cap.read()

    # Webcam
    img = cv2.resize(img, (805, 605))
    background[290:290+605, 44:44+805] = img

    # Mode image - 10% bigger
    modeImg = cv2.resize(imagemodelist[3], (568, 990))
    background[60:60+990, 880:880+568] = modeImg

    cv2.imshow("Face Attendance", background)

    cv2.waitKey(1)