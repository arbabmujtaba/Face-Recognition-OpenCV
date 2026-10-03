import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("face-attendence-3e008-firebase-adminsdk-fbsvc-b3f5e0c51c.json")
firebase_admin.initialize_app(cred,{
    "databaseURL":"https://face-attendence-3e008-default-rtdb.firebaseio.com/",
    "StorageBucket":"face-attendence-3e008.appspot.com"

})
ref = db.reference('Students')
data = {
        "192212": {
            "name": "Arbab Mujtaba",
            "major": "Computer Engineering",
            "starting_year": 2023,
            "total_attendance": 6,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2026-10-02 00:00:00"
        },
        "963852": {
            "name": "ElonMusk",
            "major": "Computer Engineering",
            "starting_year": 2023,
            "total_attendance": 3,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2026-10-09 00:00:00"
        },        
        "852741": {
            "name": "SydneySwneeny",
            "major": "Hollywood",
            "starting_year": 2023,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2026-10-05 00:00:00"
        }
    }
for key,value in data.items():
    ref.child(key).set(value)