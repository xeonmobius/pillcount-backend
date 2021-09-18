# Pillcount


## Description
### What the application does

This app counts pills for me using a custom built YoloV5s model trained on synthetic data I created from the FDA pill image database.

### Tech Stack Choice

I initial went with flutter to create the app but I ran into many issues such as lack of mature library support (no offical Tflite support yet) and not a lot of tutorials. I decided to switch over to React Native as it had better documentation and more mature libraries.

I choose YoloV5 over YoloV4 because I found the documentation easier to use and the model easier to convert to other models. But this project can easily add many different Yolo models.

### What was your motivation?
I wanted to create an app that could count pills for me. This would reduce the amount of time I would need to do inventory which includes a pharmacist to manually count every pill. A quick pharmacist can count ~5 pills/s, maybe even faster.

### Why did you build this app?
I wanted to be more efficient with my time. Inventory can take a lot of time especially when there are >10,000 pills that need to be counted.

### What problem does it solve?
I aimed to have the app count the pills for me. If the app can count 100 pills/s that would be an large boost to my time saving.

### What did you learn?
Mobile yolo models are really hard to port over. I experimented with yolov4 and tflite with flutter intially and ran into a lot of dead ends since there was lack of documentation or tutorials that repesented my use case. I eventually decided to use React Native and PyTorch since the documentation was easier to understand.

<br> 

## How to Install the Project

### Frontend

Make sure to edit the api constant in the [CameraScreen](https://github.com/xeonmobius/pillcount-ts-rn/blob/master/src/screens/CameraScreen.tsx) to point to address the backend is at.

Make sure to be in the frontend directory and run:
```console
npm start
```

### Backend

#### Using Docker

You can pull the docker image by the follow command:

```console
docker pull shannonchow93/pillcount:latest
```

Or build it from the docker file (make sure your console/terminal/shell is in the backend directory):

```console
docker build -t pillcount .
```

Then run the image by:

```console
docker run -d -p 5000:5000 pillcount
```

#### Using Flask Debug Server

If you wish to launch the backend through the flask debug server, just simply run the following:

```console
python main.py
```

Both methods will launch the app on localhost:5000

<br>

## Live demo

Download the apk from this [link](https://drive.google.com/file/d/1EMOheOdhQ5o0o057PNmkOpVMHJOerRoL/view?usp=sharing) and start counting!

<br>

## Features to Implement
- Faster Device inference on IOS and Android
- Imporove the model accuracy
- Implement the newer Yolo-R model

<Br>

## License

GNU GPLv3
