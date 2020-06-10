#!/bin/python

import os
from flask import Flask, Response, request, abort, render_template_string, send_from_directory
from flask import Flask, request, redirect, url_for, render_template
from PIL import Image
from io import StringIO

app = Flask(__name__, static_url_path='', static_folder='resources')

WIDTH = 1000
HEIGHT = 800

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title></title>
    <meta charset="utf-8" />
    <style>
body {
    margin: 0;
    background-color: #333;
}
.image {
    display: block;
    margin: 2em auto;
    background-color: #444;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
img {
    display: block;
}
    </style>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

</head>
<body>
    {% for image in images %}
        <img src="{{ image.url }}">
    {% endfor %}
</body>
'''

@app.route('/display/<filename>')
def display_image(filename):
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #print('Filename:: ', full_filename)
    url_file = url_for('static', filename='tgif-qa/video/' + filename)
    print('Resource:: ', url_file)
    return redirect(url_file, code=301)

@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='GIF')
        return Response(io.getvalue(), mimetype='image/gif')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)

@app.route('/')
def index():
    images = []
    for root, dirs, files in os.walk('resources/tgif-qa/video'):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.gif'):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0*w/h
            if aspect > 1.0*WIDTH/HEIGHT:
                width = min(w, WIDTH)
                height = width/aspect
            else:
                height = min(h, HEIGHT)
                width = height*aspect
            
            print('IMAGE:: ', filename)
            name = filename.split('/')[-1]
            images.append({
                'width': int(width),
                'height': int(height),
                'url':  '/display/' + name
            })

    return render_template_string(TEMPLATE, **{
        'images': images
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5014, debug=True)