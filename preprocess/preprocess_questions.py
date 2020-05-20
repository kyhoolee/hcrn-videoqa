import argparse
import numpy as np
import os

from preprocess.datautils import tgif_qa
from preprocess.datautils import msrvtt_qa
from preprocess.datautils import msvd_qa


# python3 preprocess/preprocess_questions.py --dataset tgif-qa-infer --question_type action --mode test
def process_question(request_id, question_type, annotate_file):
    parser = argparse.ArgumentParser()
    global args
    args = parser.parse_args()

    args.dataset = 'tgif-qa-infer'
    args.answer_top = 4000
    args.mode = 'test'
    args.question_type = question_type #['frameqa', 'action', 'transition', 'count', 'none']
    args.seed = 666
    np.random.seed(args.seed)


    
    args.annotation_file = annotate_file
    args.output_pt = 'data/tgif-qa-infer/{}/tgif-qa-infer_{}_{}_questions_' + str(request_id) + '.pt'
    args.vocab_json = 'data/tgif-qa-infer/{}/tgif-qa-infer_{}_vocab.json'

    if args.question_type in ['frameqa', 'count']:
        tgif_qa.process_questions_openended(args)
    else:
        tgif_qa.process_questions_mulchoices(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'tgif-qa-infer', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)


    if args.dataset == 'tgif-qa-infer':
        args.annotation_file = '/home/kylee/work/projects/5_vqa/dataset/tgif-qa/infer_{}_question.csv'
        args.output_pt = 'data/tgif-qa-infer/{}/tgif-qa-infer_{}_{}_questions.pt'
        args.vocab_json = 'data/tgif-qa-infer/{}/tgif-qa-infer_{}_vocab.json'
        # check if data folder exists
        if not os.path.exists('data/tgif-qa-infer/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa-infer/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)

    elif args.dataset == 'tgif-qa':
        # /home/kylee/work/projects/5_vqa/dataset/tgif-qa/Train_frameqa_question.csv
        args.annotation_file = '/home/kylee/work/projects/5_vqa/dataset/tgif-qa/{}_{}_question.csv'
        args.output_pt = 'data/tgif-qa/{}/tgif-qa_{}_{}_questions.pt'
        args.vocab_json = 'data/tgif-qa/{}/tgif-qa_{}_vocab.json'
        # check if data folder exists
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)

            
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msrvtt/annotations/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msvd/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)