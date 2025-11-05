import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("SnapCam Filter")
app.update_idletasks() 
window_width = 900
window_height = 600


screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()


x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)


app.geometry(f"{window_width}x{window_height}+{x}+{y}")

current_filter = "dog_nose"
is_running = False

filters = {
    "cat_ears_nose": cv2.imread("filters/cat_ears_nose.png", cv2.IMREAD_UNCHANGED),
    "crown_roses": cv2.imread("filters/crown_roses.png", cv2.IMREAD_UNCHANGED),
    "dog_ears": cv2.imread("filters/dog_ears.png", cv2.IMREAD_UNCHANGED),
    "rabbit_ears_nose": cv2.imread("filters/rabbit_ears_nose.png", cv2.IMREAD_UNCHANGED),
    "strange_eyes": cv2.imread("filters/strange_eyes.png", cv2.IMREAD_UNCHANGED),
    "sunglasses": cv2.imread("filters/sunglasses.png", cv2.IMREAD_UNCHANGED)
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

label = ctk.CTkLabel(app, text="")
label.pack(padx=20, pady=20)

def overlay_filter(frame, filter_img, x, y, w, h):
    if filter_img is None:
        return frame
    h_frame, w_frame, _ = frame.shape
    x1, x2 = max(0, x - w // 2), min(w_frame, x + w // 2)
    y1, y2 = max(0, y - h // 2), min(h_frame, y + h // 2)
    filter_resized = cv2.resize(filter_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    alpha = filter_resized[:, :, 3] / 255.0
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (1 - alpha) * frame[y1:y2, x1:x2, c] + alpha * filter_resized[:, :, c]
    return frame

def run_camera():
    global is_running, current_filter
    cap = cv2.VideoCapture(0)
    while is_running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                if current_filter == "cat_ears_nose":
                    nose = face_landmarks.landmark[1]
                    forehead = face_landmarks.landmark[10]
                    x = int(nose.x * w)
                    y = int((nose.y + forehead.y)/2 * h - 20)
                    scale_factor = 1.5 
                    width = int(abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w * scale_factor)
                    height = int(width * filters[current_filter].shape[0] / filters[current_filter].shape[1])
                    frame = overlay_filter(frame, filters[current_filter], x, y, width, height)

                elif current_filter == "rabbit_ears_nose":
                    nose = face_landmarks.landmark[1]
                    forehead = face_landmarks.landmark[10]
                    x = int(nose.x * w)
                    y = int((nose.y + forehead.y)/2 * h - 20)
                    scale_factor = 2.4
                    width = int(abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w * scale_factor)
                    height = int(width * filters[current_filter].shape[0] / filters[current_filter].shape[1])
                    frame = overlay_filter(frame, filters[current_filter], x, y, width, height)

                elif current_filter == "crown_roses":
                    left_temple = face_landmarks.landmark[127]
                    right_temple = face_landmarks.landmark[356]
                    forehead = face_landmarks.landmark[10]
                    x = int((left_temple.x + right_temple.x)/2 * w)
                    y = int(forehead.y * h - 30)
                    scale_factor = 1.5 
                    width = int(abs(right_temple.x - left_temple.x) * w * scale_factor)
                    height = int(width * filters[current_filter].shape[0] / filters[current_filter].shape[1])
                    frame = overlay_filter(frame, filters[current_filter], x, y, width, height)

                elif current_filter == "strange_eyes":
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    x = int((left_eye.x + right_eye.x)/2 * w)
                    y = int((left_eye.y + right_eye.y)/2 * h)
                    scale_factor = 1.8 
                    width = int(abs(right_eye.x - left_eye.x) * w * scale_factor)
                    height = int(width * filters[current_filter].shape[0] / filters[current_filter].shape[1])
                    frame = overlay_filter(frame, filters[current_filter], x, y, width, height)

                elif current_filter == "sunglasses":
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    x = int((left_eye.x + right_eye.x)/2 * w)
                    y = int((left_eye.y + right_eye.y)/2 * h)
                    scale_factor = 1.8
                    width = int(abs(right_eye.x - left_eye.x) * w * scale_factor)
                    height = int(width * filters[current_filter].shape[0] / filters[current_filter].shape[1])
                    frame = overlay_filter(frame, filters[current_filter], x, y, width, height)

                elif current_filter == "dog_ears":
                    left_eyebrow = face_landmarks.landmark[70]
                    right_eyebrow = face_landmarks.landmark[300]
                    x = int((left_eyebrow.x + right_eyebrow.x)/2 * w)
                    y = int(min(left_eyebrow.y, right_eyebrow.y) * h - 50)
                    scale_factor = 2
                    width = int(abs(right_eyebrow.x - left_eyebrow.x) * w * scale_factor)
                    height = int(width * filters[current_filter].shape[0] / filters[current_filter].shape[1])
                    frame = overlay_filter(frame, filters[current_filter], x, y, width, height)


        display_frame = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    cap.release()

def start_camera():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=run_camera, daemon=True).start()

def stop_camera():
    global is_running
    is_running = False
    label.configure(image='')

def set_filter(name):
    global current_filter
    current_filter = name

frame_buttons = ctk.CTkFrame(app)
frame_buttons.pack(pady=20)

ctk.CTkButton(frame_buttons, text="Start Camera", command=start_camera).grid(row=0, column=0, padx=10)
ctk.CTkButton(frame_buttons, text="Stop Camera", command=stop_camera).grid(row=0, column=1, padx=10)
ctk.CTkButton(frame_buttons, text="Remove Filter", command=lambda: set_filter(None)).grid(row=0, column=2, padx=10)
ctk.CTkButton(frame_buttons, text="Cat Ears Nose", command=lambda: set_filter("cat_ears_nose")).grid(row=1, column=0, pady=10)
ctk.CTkButton(frame_buttons, text="Crown Roses", command=lambda: set_filter("crown_roses")).grid(row=1, column=1, pady=10)
ctk.CTkButton(frame_buttons, text="Sunglasses", command=lambda: set_filter("sunglasses")).grid(row=1, column=2, pady=10)
ctk.CTkButton(frame_buttons, text="Dog Ears", command=lambda: set_filter("dog_ears")).grid(row=2, column=0, pady=10)
ctk.CTkButton(frame_buttons, text="Rabbit Ears Nose", command=lambda: set_filter("rabbit_ears_nose")).grid(row=2, column=1, pady=10)
ctk.CTkButton(frame_buttons, text="Strange Eyes", command=lambda: set_filter("strange_eyes")).grid(row=2, column=2, pady=10)

app.mainloop()
