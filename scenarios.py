import mediapipe as mp
import cv2
import time
from ultralytics import YOLO

########## MediaPipe on webcam ##########
# def detect_faces():
#     # Initialize MediaPipe Face Detection
#     mp_face_detection = mp.solutions.face_detection
#     mp_drawing = mp.solutions.drawing_utils
#     face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

#     # Use webcam (0 is usually the built-in webcam, 1 would be an external webcam if connected)
#     cap = cv2.VideoCapture(0)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame
#         results = face_detection.process(rgb_frame)

#         # Draw detections
#         if results.detections:
#             for detection in results.detections:
#                 mp_drawing.draw_detection(frame, detection)

#         # Display the frame
#         cv2.imshow('Face Detection', frame)
        
#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()




########## MediaPipe on video clips ##########
# def detect_faces(video_path):
#     # Initialize MediaPipe Face Detection
#     mp_face_detection = mp.solutions.face_detection
#     mp_drawing = mp.solutions.drawing_utils
#     face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

#     mp_drawing = mp.solutions.drawing_utils

#     # Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     # Get the original video's FPS
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_time = 1/45  # time per frame in seconds

#     while cap.isOpened():
#         start_time = time.time()  # Start time for this frame
        
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resize frame
#         scale_percent = 35
#         width = int(frame.shape[1] * scale_percent / 100)
#         height = int(frame.shape[0] * scale_percent / 100)
#         dim = (width, height)
#         frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame
#         results = face_detection.process(rgb_frame)

#         # Draw detections
#         if results.detections:
#             for detection in results.detections:
#                 mp_drawing.draw_detection(frame, detection)

#         # Display the frame
#         cv2.imshow('Face Detection', frame)
        
#         # Calculate how long to wait
#         processing_time = time.time() - start_time
#         wait_time = max(1, int((frame_time - processing_time) * 1000))
        
#         if cv2.waitKey(wait_time) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



# if __name__ == '__main__':
#     detect_faces_mediapipe('./videos/3.mp4')




########## PURE YOLO (without Supervision) ##########
# def detect_faces(video_path):
#     # Load YOLOv8 model
#     model = YOLO('yolov8n.pt')
    
#     cap = cv2.VideoCapture(video_path)
#     # frame_time = 1/30  # 30 FPS
    
#     while cap.isOpened():
#         start_time = time.time()
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resizing frame for better performance
#         scale_percent = 40
#         width = int(frame.shape[1] * scale_percent / 100)
#         height = int(frame.shape[0] * scale_percent / 100)
#         frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

#         # Detect people (class 0 is person)
#         results = model(frame, classes=[0])  # only detect people
        
#         # Draw the results on the frame
#         annotated_frame = results[0].plot()
        
#         # Display the frame
#         cv2.imshow('Person Detection', annotated_frame)
        
#         # # Control FPS
#         # processing_time = time.time() - start_time
#         # wait_time = max(1, int((frame_time - processing_time) * 1000))
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_faces()