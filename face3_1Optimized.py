import face_recognition
import cv2


known_face_encodings = []
known_face_names = []



'''AQUI COLOCAN LAS RUTAS DE IMAGENES CONOCIDAS Y ALADO EL NOMBRE ASOCIADO A LA PERSONA'''
images_info = [
    ("fotos/obama.jpeg", "Obama"),
    ("fotos/persona1", "NombrePersona1")
]


for image_path, name in images_info:
    image = cv2.imread(image_path)
    face_loc = face_recognition.face_locations(image)[0]
    face_encoding = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(3, 1920) 
cap.set(4, 1080)  

process_this_frame = True
frame_count = 0
process_every_n_frames = 5  

last_known_names = []
last_known_locations = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

  
    if frame_count % process_every_n_frames == 0:
       
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        last_known_names = []
        last_known_locations = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            last_known_names.append(name)
            last_known_locations.append(face_location)

   
    for name, (top, right, bottom, left) in zip(last_known_names, last_known_locations):
        top, right, bottom, left = [val * 4 for val in (top, right, bottom, left)] 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    frame_count += 1

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27: 
        break

cap.release()
cv2.destroyAllWindows()
