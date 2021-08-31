from flask import Flask, request
from flask_cors import cross_origin
import re
import tensorflow as tf
import numpy as np
import detect

app = Flask(__name__)

def base64_to_url_safe(base64):
    """
    Function removes the data... which is not needed for the
    actual base64 string and then replaces all / and + with url
    save characters
    """
    s = re.sub('^data:image/.+;base64,', '', base64)
    s = s.replace(r"/", "_").replace('+', '-')
    return s

def base64_to_html_base_64(base64):
    """
    Adds the "data:image/jpeg;base64," to the base64 string so
    it can be properly shown in the image tag.
    """
    s = "data:image/jpeg;base64," + base64
    return s

# End point to recieve images
@app.route('/api/yolov4', methods=['POST'])
@cross_origin()
def yolov4():
    return "Not Implemented"


@app.route('/api/yolov5/detect', methods=['POST'])
@cross_origin()
def yolov5():
    """
    API endpoint for the yoloV5 model to detect pills.
    Takes in a picture in base64 format and returns a new
    base64 image with the bounding box and predictions.
    """
    image_base64 = base64_to_url_safe(request.json["image"])
    
    decoded_b64 = tf.io.decode_base64(image_base64)
    image = tf.image.decode_jpeg(decoded_b64)
    
    np_img = np.asarray(image)

    result = detect.run(source=np_img)
    result['image'] = base64_to_html_base_64(result['image'])
    
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)