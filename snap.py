from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
from Music_lift.music_model import MusicMoodClassifier
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+" * 50, "loadin gmmodel")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/youtube', methods=["GET", "POST"])
def youtube():
    return render_template('youtube.html')


@app.route('/choose_singer', methods=["POST"])
def choose_singer():
    info['language'] = request.form['language']
    print(info)
    return render_template('choose_singer.html', data=info['language'])


@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    info['singer'] = request.form['singer']

    found = False

    cap = cv2.VideoCapture(0)
    while not found:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))

    roi = roi / 255.0

    roi = np.reshape(roi, (1, 48, 48, 1))

    prediction = model.predict(roi)

    print(prediction)

    prediction = np.argmax(prediction)
    prediction = label_map[prediction]

    cap.release()

    link = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction}+{info['language']}+song"
    webbrowser.open(link)

    return render_template("emotion_detect.html", data=prediction, link=link)


@app.route('/song_recommend', methods=["POST", "GET"])
def song_recommend():
    found = False

    cap = cv2.VideoCapture(0)
    while not found:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))

    roi = roi / 255.0

    roi = np.reshape(roi, (1, 48, 48, 1))

    prediction = model.predict(roi)

    print(prediction)

    prediction = np.argmax(prediction)
    prediction = label_map[prediction]

    cap.release()
    rf = MusicMoodClassifier()
    if prediction == "Anger":
        label = 0 # if anger detected suggest calm music
    elif prediction == "Neutral":
        label = 2 # if Neutral detected suggest happy music
    elif prediction == "Fear":
        label = 0 # if fear detected suggest calm music
    elif prediction == "Happy":
        label = 2 # if Happy detected suggest happy music
    elif prediction == "Sad":
        label = 3 # if sad detected suggest sad music
    elif prediction == "Surprise":
        label = 1 # if surprise detected suggest energetic music
    else:
        print("No emotion detected")
    prob = rf.getTypicalTracks(label)
    return render_template('song_recommend.html', emotion=prediction, Music=prob[0], URL=prob[1], Album=prob[2])



if __name__ == "__main__":
    app.run(debug=True)