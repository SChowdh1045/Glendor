import cv2
from ultralytics import YOLO
import supervision as sv
import time

def detect_people(video_path):
    # Load YOLO model
    model = YOLO('./models/yolov8n.pt')
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    # frame_time = 1/30
    
    # Create BoxAnnotator instance
    box_annotator = sv.BoxAnnotator(
        color=sv.Color(r=200, g=50, b=200),
        thickness=2
    )

    while cap.isOpened():
        # start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        scale_percent = 40
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Detect people
        results = model(frame, classes=[0])[0]
        
        # Convert detections to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Annotate the frame
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections
        )
        
        # Display detection count
        detection_count = len(detections)
        cv2.putText(
            frame, 
            f"People detected: {detection_count}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        # Display the frame
        cv2.imshow('Person Detection', frame)
        
        # # Control FPS
        # processing_time = time.time() - start_time
        # wait_time = max(1, int((frame_time - processing_time) * 1000))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_people('./videos/3.mp4')