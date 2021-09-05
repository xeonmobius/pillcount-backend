from flask import Flask, request
from flask_cors import cross_origin, CORS
import torch
import cv2
import util.base64_util as b64_util


app = Flask(__name__)
cors = CORS(app)

# Load our yolov5 model ++++ ALWAYS MAKE FORCE RELOAD TRUE OTHER WISE THE MODEL LOADING WILL FAIL!!!!!!
print("Loading Yolov5 Model")
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="model/best.pt", force_reload=True
)
model.eval()

# End point to recieve images
@app.route("/")
def index():
    return "Pillcount server is running!"


# End point for yolo to recieve images
@app.route("/api/yolov5/detect", methods=["POST"])
@cross_origin()
def yolov5():
    """
    API endpoint for the yoloV5 model to detect pills.
    Takes in a picture in base64 format and returns a new
    base64 image with the bounding box and predictions.
    """

    print("YOLO V5 Detection requested")
    np_img = b64_util.base64_to_np_img(request.json["image"])

    # inference on the picture using our custom model
    result = model(np_img)

    # filter the results to just > 85% confidence
    confidence_filter = result.pandas().xyxy[0]["confidence"] > 0.85
    filtered_result = result.pandas().xyxy[0][confidence_filter]

    nums = len(filtered_result)

    for row in filtered_result.iterrows():
        x_min, y_min, x_max, y_max = row[1][:4].astype(int)

        # Draw rectangles
        cv2.rectangle(np_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # Convert our image back to base64
    b64s = b64_util.np_img_to_base64(np_img)

    print("YOLO V5 Detection successful")
    return {"image": b64s, "predictions": f"{nums}"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
