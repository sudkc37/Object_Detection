import cv2
import torch
import supervision as sv
import easyocr
from transformers import DetrImageProcessor, DetrForObjectDetection

def load_model(model_path, device):
    model = DetrForObjectDetection.from_pretrained(model_path)
    return model.to(device)

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not access camera')
        exit()
    return cap

def process_frame(frame, model, image_processor, device, box_annot, img_reader):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        input_tensor = image_processor(images=img, return_tensors='pt').to(device)
        outputs = model(**input_tensor)
        
        target_size = torch.tensor([img.shape[:2]]).to(device)
        results = image_processor.post_process_object_detection(
            outputs=outputs, threshold=0.85, target_sizes=target_size
        )[0]

        detections = sv.Detections.from_transformers(transformers_results=results)
        
        for box in detections.xyxy:
            x_min, y_min, x_max, y_max = map(int, box)
            plate_img = frame[y_min:y_max, x_min:x_max]
            plate_text = img_reader.readtext(plate_img, detail=0)
            
            if plate_text:
                detected_text = plate_text[1]
                print(f"Detected Plate: {detected_text}")
                text_position = (x_min, max(y_min - 10, 20))
                cv2.putText(
                    frame, f"Plate: {detected_text}", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )
        
        return box_annot.annotate(scene=frame, detections=detections)

def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    model = load_model('/Users/sudipkhadka/Desktop/Object_Detection/Research/trained_model', DEVICE)
    box_annot = sv.BoxAnnotator()
    img_reader = easyocr.Reader(['en'], gpu=False)
    cap = initialize_camera()

    print("Searching for License Plates........ Press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Could not read frame')
                break
            
            frame = process_frame(frame, model, image_processor, DEVICE, box_annot, img_reader)
            cv2.imshow('License Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
             
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('Camera Released and all windows closed')

if __name__ == "__main__":
    main()
