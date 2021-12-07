# -*- coding: utf-8 -*-
import base64
import os
import uuid
import cv2
import torch

from flask import Flask, render_template, redirect, url_for, request, send_file
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from dataloader.dataloader import ImageNetDataset
from eval.eval import HotDogNotHotDogInfer

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder='./static')
app.config['SECRET_KEY'] = 'secret'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        for f in request.files.getlist('photo'):
            filename = uuid.uuid4().hex
            photos.save(f, name=filename + '.')
        success = True

        prediction = infer(os.path.join(basedir, 'uploads', filename + '.' + form.data['photo'].mimetype.split('/')[1]))

    else:
        success = False
        prediction = ''
    return render_template('index.html', form=form, success=success, prediction=prediction)


def infer(filename):
    model_tag = 'epoch_model_256'
    infer_model = HotDogNotHotDogInfer()
    infer_model.load_model(model_tag)

    query = ImageNetDataset.load_img(filename)

    prediction = infer_model.infer(query)

    labels = {0: 'not hotdog', 1: 'hotdog'}

    return labels[torch.argmax(prediction).item()]


# File Manager

@app.route('/manage')
def manage_file():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    return render_template('manager.html', files_list=files_list)


@app.route('/open/<filename>')
def open_file(filename):
    img = cv2.imread('uploads/' + filename)
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    img_src = base64.b64encode(im_bytes)
    # return send_file(file_url, mimetype='image/png')
    return render_template('explorer.html', img_src=img_src)


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_file'))


if __name__ == '__main__':
    app.run(debug=True)
