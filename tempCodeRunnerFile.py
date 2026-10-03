if totalAttendance is not None and counter <=10 and 0<frame_count<60:
        cv2.putText(background, str(studentinfo['total_attendance']),(955,190), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['name']),(965,700), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['major']),(1147,852), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(background, str(id),(1147,765), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['standing']),(1020,984), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['year']),(1172,984), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 1)
        cv2.putText(background, str(studentinfo['starting_year']),(1300,976), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0,0,0), 1)
    cv2.imshow("Face Attendance", background)