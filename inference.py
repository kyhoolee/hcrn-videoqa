import validate 
from dotmap import DotMap
import preprocess.preprocess_features as preprocess_features
import preprocess.preprocess_questions as preprocess_questions
import preprocess.datautils.tgif_qa as tgif_qa

import os
import pandas as pd
import json
from preprocess.datautils import utils
import nltk

import pickle
import numpy as np






def load_video_paths_by_request(video_dir, file_path):
    ''' Load a list of (path,image_id tuples).'''
    input_paths = []
    annotation = pd.read_csv(file_path, delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(video_dir, gif)
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths


def sample_process_all():
    request_id = 'req1'
    question_type = 'count'


    annotation_file = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/csv/infer_{}_question_' + str(request_id) + '.csv'
    video_dir = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/video/'
    question_model = 'expTGIF-QACount'

    process_all(request_id, question_type, question_model, annotation_file, video_dir)


def sample_process_cache():
    request_id = 'f953d88f'
    question_type = 'action'
    annotation_file = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/csv/infer_{}_question_' + str(request_id) + '.csv'
    video_dir = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/video/'
    question_model = 'expTGIF-QAAction'

    process_all(request_id, question_type, question_model, annotation_file, video_dir, is_cache=True)


def sample_process_cache_frameqa():
    request_id = 'e6118db3'
    question_type = 'frameqa'
    annotation_file = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/csv/infer_{}_question_' + str(request_id) + '.csv'
    video_dir = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/video/'
    question_model = 'expTGIF-QAFrameQA'
    video_id = 'tumblr_no73q2fm0I1uuf348o1_250'

    process_all_cache(request_id, question_type, question_model, annotation_file, video_dir, video_id)


def sample_process_cache_count():
    request_id = 'cb89a145'
    question_type = 'count'
    annotation_file = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/csv/infer_{}_question_' + str(request_id) + '.csv'
    video_dir = '/home/kylee/work/projects/5_vqa/hcrn-videoqa/resources/tgif-qa/video/'
    question_model = 'expTGIF-QACount'
    video_id = 'tumblr_ndar2w45c51t7jda2o1_250'

    process_all_cache(request_id, question_type, question_model, annotation_file, video_dir, video_id)


def process_all(request_id, question_type, question_model, annotation_file, video_dir, is_cache=False):
        
    video_paths = load_video_paths_by_request(video_dir, annotation_file.format(question_type))



    if not is_cache:
        preprocess_features.preprocess_infer_motion(video_paths, request_id, question_type)
        print('Done motion')
        preprocess_features.preprocess_infer_appearance(video_paths, request_id, question_type)
        print('Done appearance')
    
    preprocess_questions.process_question(request_id, question_type, annotation_file)
    print('Done question')
    result = process_final(request_id, question_type, question_model)

    return result

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_all_cache(request_id, question_type, question_model, annotation_file, video_dir, video_id):
    make_folder('data/tgif-qa-infer/video')
        
    video_paths = load_video_paths_by_request(video_dir, annotation_file.format(question_type))

    preprocess_features.preprocess_infer_motion_cache(video_paths, request_id, question_type, video_id)
    print('Done motion')
    preprocess_features.preprocess_infer_appearance_cache(video_paths, request_id, question_type, video_id)
    print('Done appearance')
    
    preprocess_questions.process_question(request_id, question_type, annotation_file)
    print('Done question')
    result = process_final_cache(request_id, question_type, question_model, video_id)

    return result


def process_final_cache(request_id, question_type, question_model, video_id):
    config = '''
    {
        "gpu_id": 1, 
        "num_workers": 4, 
        "multi_gpus": false, 
        "seed": 666, 
        "train": {
            "restore": false, 
            "lr": 0.0001, 
            "batch_size": 1, 
            "max_epochs": 25, 
            "vision_dim": 2048, 
            "word_dim": 300, 
            "module_dim": 512, 
            "train_num": 0, 
            "glove": true, 
            "k_max_frame_level": 16, 
            "k_max_clip_level": 8, 
            "spl_resolution": 1
        }, 
        "val": {
            "flag": true, 
            "val_num": 0
        }, 
        "test": {
            "test_num": 0, 
            "write_preds": true
        }, 
        "dataset": {
            "name": "tgif-qa-infer", 
            "question_type": "%question_type%", 
            "data_dir": "data/tgif-qa-infer/%question_type%", 
            "video_dir": "data/tgif-qa-infer/video",
            "appearance_feat": "{}_appearance_video_%video_id%.h5", 
            "motion_feat": "{}_motion_video_%video_id%.h5", 
            "vocab_json": "{}_{}_vocab.json", 
            "train_question_pt": "{}_{}_train_questions.pt", 
            "val_question_pt": "{}_{}_val_questions.pt", 
            "test_question_pt": "{}_{}_test_questions_%req%.pt", 
            "save_dir": "results/%question_model%/"
        }, 
        "exp_name": "%question_model%"
    }
    '''
    config = config.replace('%video_id%', video_id)
    config = config.replace('%req%', request_id)
    config = config.replace('%question_type%', question_type)
    # expTGIF-QAAction
    config = config.replace('%question_model%', question_model)

    import json
    jsonDict = json.loads(config)
    data = DotMap(jsonDict)

    return validate.process_final(data)



def process_final(request_id, question_type, question_model):
    config = '''
    {
        "gpu_id": 1, 
        "num_workers": 4, 
        "multi_gpus": false, 
        "seed": 666, 
        "train": {
            "restore": false, 
            "lr": 0.0001, 
            "batch_size": 1, 
            "max_epochs": 25, 
            "vision_dim": 2048, 
            "word_dim": 300, 
            "module_dim": 512, 
            "train_num": 0, 
            "glove": true, 
            "k_max_frame_level": 16, 
            "k_max_clip_level": 8, 
            "spl_resolution": 1
        }, 
        "val": {
            "flag": true, 
            "val_num": 0
        }, 
        "test": {
            "test_num": 0, 
            "write_preds": true
        }, 
        "dataset": {
            "name": "tgif-qa-infer", 
            "question_type": "%question_type%", 
            "data_dir": "data/tgif-qa-infer/%question_type%", 
            "appearance_feat": "{}_{}_appearance_feat_%req%.h5", 
            "motion_feat": "{}_{}_motion_feat_%req%.h5", 
            "vocab_json": "{}_{}_vocab.json", 
            "train_question_pt": "{}_{}_train_questions.pt", 
            "val_question_pt": "{}_{}_val_questions.pt", 
            "test_question_pt": "{}_{}_test_questions_%req%.pt", 
            "save_dir": "results/%question_model%/"
        }, 
        "exp_name": "%question_model%"
    }
    '''
    
    config = config.replace('%req%', request_id)
    config = config.replace('%question_type%', question_type)
    # expTGIF-QAAction
    config = config.replace('%question_model%', question_model)

    import json
    jsonDict = json.loads(config)
    data = DotMap(jsonDict)

    return validate.process_final(data)


if __name__ == '__main__':
    # process_final()
    # sample_process_all()
    # sample_process_cache_count()
    sample_process_cache_frameqa()
    