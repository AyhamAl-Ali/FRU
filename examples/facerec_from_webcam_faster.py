import face_recognition
import cv2
import numpy as np
from datetime import datetime


#!! Facial Recognition for Universities (FRU)
#!! Artificial Intelligence Project
#!! 
#!! Team: Ayham Al-Ali & Ali-Al-Qaisi
#!! 
#!! Features:
#!! 1- Detect if student is Present, Late or Absent (Absent by default) - ('Excused' status is manually assigned by Doctor)
#!! 2- After 5 minutes of when the application starts (in-other-words when the class begins) the student will be marked as Late
#!! 3- Before the first 5 minutes the studednt will be marked as Present
#!! 4- Logs the enter time of each student
#!! 


# This is a demo includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.







#
# Options
#
late_after_time = 5 # by minutes
















#
# Code - DO NOT EDIT UNLESS YOU KNOW WHAT YOU'RE DOING
#

current_app_time = now = datetime.now()


#! Get a reference to webcam #0 (the default one)
#! video_capture = cv2.VideoCapture(0)

#! Ghangeable based on the external camera I use since my default webcam (laptop's) is broken
#! Check the number by checking available `video` in `/dev/videoX` where X is the number
video_capture = cv2.VideoCapture(2)

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

def addKnownFace(file_name, person_name):
    # image_file = face_recognition.load_image_file(file_name)
    face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(file_name))[0]

    known_face_encodings.append(face_encoding)
    known_face_names.append(person_name)


known_faces_folder_path = "./known_people/"

# Add faces here
addKnownFace(known_faces_folder_path + "Ayham.jpg", "Ayham")
addKnownFace(known_faces_folder_path + "Fahad.jpg", "Fahad")
addKnownFace(known_faces_folder_path + "Miar.jpg", "Miar")
addKnownFace(known_faces_folder_path + "Ammar.jpg", "Ammar")
addKnownFace(known_faces_folder_path + "Zaid.jpg", "Zaid")


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # face_status = []

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "0x0" # ;)

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # if (datetime.now() - current_app_time).total_seconds() > late_after_time * 60:
            #     face_status.append("Late")
            # else:
            #     face_status.append("Present")

            if (datetime.now() - current_app_time).total_seconds() > late_after_time * 60:
                name += ", s: Late"
            elif name != "0x0":
                name += ", s: Present"
            else:
                name += ", s: 0x0"

            face_names.append(name)


    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        status = name.split(", s: ")[1]
        name = name.split(", s: ")[0]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Draw a second label with the status (Late, Present)
        status_color = []
        if status == "Present":
            status_color = [0, 200, 0]
        elif status == "Late":
            status_color = [255, 165, 0]
        else:
            status_color = [224, 73, 62]

        cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (status_color[0], status_color[1], status_color[2]), cv2.FILLED)
        cv2.putText(frame, status, (left + 6, bottom + 35 - 6), font, 1.0, (255, 255, 255), 1)


    # Display the resulting image
    cv2.imshow('FRU - AI Project - AA Team', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
