import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import csv
import os
import uuid 
import inference
import json
import gc


_dir = os.path.abspath(os.path.dirname(__file__))


UPLOAD_FOLDER = _dir + '/resources/tgif-qa/video/'
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['gif'])

BASE_PATH = '/videoqa'

question_type_model = {
    'action': 'expTGIF-QAAction',
    'frameqa': 'expTGIF-QAFrameQA',
    'transition': 'expTGIF-QATransition',
    'count': 'expTGIF-QACount',    
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def write_request(data_dict, question_type, request_id):


    csv_filename = _dir + '/resources/tgif-qa/csv/infer_{}_question_{}.csv'
    csv_filename = csv_filename.format(question_type, request_id)
    
    with open(csv_filename, 'w') as f:
        writer = csv.DictWriter(
            f, 
            delimiter='\t',
            # gif_name	question	a1	a2	a3	a4	a5	answer	vid_id	key
            fieldnames=['gif_name', 'question', 'a1', 'a2', 'a3', 'a4', 'a5', 'answer', 'vid_id', 'key']
            )
        writer.writeheader()
        writer.writerows(data_dict)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def gen_key():
    return str(uuid.uuid4().hex[:8])

@app.route(BASE_PATH + "/process", methods=['GET', 'POST'])
def process():
    gc.collect()
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            question = request.form['question']
            question_type = request.form['question_type']
            
            answer = request.form['answer']
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            unique_key = gen_key()

            data_dict = []
            if question_type == 'action' or question_type == 'transition':
                a1 = request.form['a1']
                a2 = request.form['a2']
                a3 = request.form['a3']
                a4 = request.form['a4']
                a5 = request.form['a5']
                data_dict = [{
                    'gif_name': str(filename),
                    'question': str(question),
                    'a1': str(a1),
                    'a2': str(a2),
                    'a3': str(a3),
                    'a4': str(a4),
                    'a5': str(a5),
                    'answer': str(answer),
                    'vid_id': str(unique_key),
                    'key': 1

                }]
            elif question_type == 'frameqa' or question_type == 'count':
                data_dict = [{
                    'gif_name': str(filename),
                    'question': str(question),
                    'answer': str(answer),
                    'vid_id': str(unique_key),
                    'key': 1

                }]

            request_id = unique_key
            # question_type = 'action'
            question_model = question_type_model[question_type] #'expTGIF-QAAction'

            print('Data_dict:: ', data_dict)
            print('question_type:: ', question_type, ' :: Question_model :: ', question_model)
            annotation_file = _dir + '/resources/tgif-qa/csv/infer_{}_question_' + request_id + '.csv'
            video_dir = UPLOAD_FOLDER

            write_request(data_dict, question_type, request_id)

            # return redirect(url_for('index'))
            result_dict = inference.process_all(request_id, question_type, question_model, annotation_file, video_dir)
            result = json.dumps(result_dict)
            result = result.replace('<UNK>', '__UNK__')
            print(result)

            
            gc.collect()
            return result
        else: 
            return 'Error file video extension - must be GIF'

@app.route(BASE_PATH + "/", methods=['GET'])
def index():
    # gif_name	question	a1	a2	a3	a4	a5	answer	vid_id	key
    return render_template('home.html')


def sample_write_csv():
    data_dict = dict({
                'gif_name': str(1),
                'question': str(2),
                'a1': str(3),
                'a2': str(4),
                'a3': str(5),
                'a4': str(6),
                'a5': str(7),
                'answer': str(8),
                'vid_id': str(9),
                'key': str(10)

            })
    
    csv_filename = _dir + '/resources/tgif-qa/csv/infer_{}_question_{}.csv'
    csv_filename = csv_filename.format('s', 'ss')
    
    with open(csv_filename, 'w') as f:
        writer = csv.DictWriter(
            f, 
            delimiter='\t',
            # gif_name	question	a1	a2	a3	a4	a5	answer	vid_id	key
            fieldnames=['gif_name', 'question', 'a1', 'a2', 'a3', 'a4', 'a5', 'answer', 'vid_id', 'key']
            )
        writer.writeheader()
        writer.writerow(data_dict)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True)
