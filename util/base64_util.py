import io
import base64
import re
import numpy as np
from PIL import Image

def base64_to_np_img(b64s):
    """
    Removes the "data..." tag so only the base64 str is left.
    Convers base64 to an np array.
    """

    data = re.sub('^data:image/.+;base64,', '', b64s) # remove data... to get just base64 string
    buf = io.BytesIO(base64.b64decode(data))
    img = Image.open(buf)
    np_img = np.asarray(img)

    return np_img

def np_img_to_base64(np_img):
    """
    Encodes the np_img matrix to base64 string
    Adds the "data:image/jpeg;base64," to the base64 string so
    it can be properly shown in the image tag.
    """

    img = Image.fromarray(np_img)
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)

    b64s = str(base64.b64encode(rawBytes.read()).decode())
    b64s = "data:image/jpeg;base64," + b64s

    return b64s