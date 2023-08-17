# https://bong-sik.tistory.com/16 (학습된 모델 적용법 참고용 예제)

import cv2
import numpy as np

# Yolo 로드( LFS로 model 업데이트가 안되서 tiny로 변경 후 push)
def detection (uploadedImage,net = cv2.dnn.readNet("model/yolov3-tiny.cfg","model/yolov3-tiny.weights")):

    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지 가져오기
    # img = cv2.imread("test/sample.jpg")
    img = cv2.imdecode(np.fromstring(uploadedImage.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # 이미지에서 특징을 잡아내고 크기를 조정 by Blob for Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # 네트워크에서 이미지를 바로 사용할 수 없기때문에 먼저 이미지를 Blob으로 변환
    net.setInput(blob)
    # outs 감지 결과. 탐지된 개체에 대한 모든 정보와 위치를 제공
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    # 임계값 0 ~ 1 사이 값, 1에 가까울수록 탐지 정확도가 높고 , 0에 가까울수록 정확도는 낮아지지만 탐지되는 물체의 수는 증가
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 노이즈 제거(같은 물체에 대한 박스가 많은것을 제거 Non maximum suppresion)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 화면에 표시하기 : Box(감지된 개체를 둘러싼 사각형의 좌표), Label(감지된 물체의 이름), Confidence(0에서 1까지의 탐지에 대한 신뢰도),
    fontFace = cv2.FONT_HERSHEY_PLAIN
    detectedLabels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detectedLabels.append(label)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), fontFace, 2, color, 2)
    return img, detectedLabels