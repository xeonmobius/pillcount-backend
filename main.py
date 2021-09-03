from flask import Flask, request
from flask_cors import cross_origin
import torch 
import cv2
import util.base64_util as b64_util


app = Flask(__name__)


# End point to recieve images
@app.route('/')
def index():
    return "Pillcount server is running!"

# End point to recieve images
@app.route('/api/yolov4', methods=['POST'])
@cross_origin()
def yolov4():
    return "Not Implemented"

# End point for yolo to recieve images
@app.route('/api/yolov5/detect', methods=['POST'])
@cross_origin()
def yolov5():
    """
    API endpoint for the yoloV5 model to detect pills.
    Takes in a picture in base64 format and returns a new
    base64 image with the bounding box and predictions.
    """

    np_img = b64_util.base64_to_np_img(request.json["image"])

    result = model(np_img) # inference on the picture using our custom model
    
    # filter the results to just > 85% confidence
    confidence_filter = result.pandas().xyxy[0]["confidence"] > 0.85
    filtered_result = result.pandas().xyxy[0][confidence_filter]

    nums = len(filtered_result)

    for row in filtered_result.iterrows():
        x_min, y_min, x_max, y_max = row[1][:4].astype(int)

        # Draw rectangles
        cv2.rectangle(np_img, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
    
    b64s = b64_util.np_img_to_base64(np_img)

    return {"image": b64s, "predictions": f"{nums}"}


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt')
    model.eval()
    app.run(host='0.0.0.0', port=5000)
    
