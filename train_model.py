from ultralytics import YOLO

#model = YOLO("../Yolo_weights/yolov8n.yaml")
#model = YOLO("yolov8n.pt")
if __name__ == '__main__':
    #model = YOLO("yolov8n.yaml").load("../Yolo_weights/yolov8n.pt")
    model = YOLO("../Yolo_weights/yolov8n.pt")
    results = model.train(data="../Datasets/sign_language_v2/data.yaml", epochs=39, imgsz=640,device=0)
    #success = YOLO("../Yolo_weights/sign_lang_v2.pt").export(format="onnx")