import torch
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored

from DataLoader import VideoQADataLoader
from utils import todevice

import model.HCRN as HCRN

from config import cfg, cfg_from_file



def validate(cfg, model, data, device, write_preds=False):
    model.eval()
    print('validating...')
    print('DATA::', len(data), type(data))
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            print('video_ids: ', video_ids)
            print('question_ids: ', question_ids)
            print('answer: ', answers)
            
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()

            print('answer: ', answers)
                
            batch_size = answers.size(0)

            logits = model(*batch_input).to(device)

            print('LOGITS:: ', type(logits), logits.size() , logits)


            if cfg.dataset.question_type in ['action', 'transition']:
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                agreeings = (preds == answers)
            elif cfg.dataset.question_type == 'count':
                answers = answers.unsqueeze(-1)
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2
            else:
                preds = logits.detach().argmax(1)
                print('FRAMEQA-PREDs:: ', preds)
                agreeings = (preds == answers)


            if write_preds:
                if cfg.dataset.question_type not in ['action', 'transition', 'count']:
                    preds = logits.argmax(1)

                if cfg.dataset.question_type in ['action', 'transition']:
                    answer_vocab = data.vocab['question_answer_idx_to_token']
                else:
                    answer_vocab = data.vocab['answer_idx_to_token']
                for predict in preds:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        all_preds.append(predict.item())
                    else:
                        print('FRAMEQA-pred.item:: ', predict.item())
                        # print('FRAMEQA-answer_vocab:: ', answer_vocab)
                        all_preds.append(answer_vocab[predict.item()])


                for gt in answers:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        gts.append(gt.item())
                    else:
                        gts.append(answer_vocab[gt.item()])

                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id)

            if cfg.dataset.question_type == 'count':
                total_acc += batch_mse.float().sum().item()
                count += answers.size(0)
            else:
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)
        acc = total_acc / count
    if not write_preds:
        return acc
    else:
        return acc, all_preds, gts, v_ids, q_ids, logits


def process_final(cfg):

    assert cfg.dataset.name in ['tgif-qa', 'tgif-qa-infer', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    if cfg.dataset.name != 'tgif-qa-infer':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    # cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    print('ckpt:: ', ckpt)
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    if cfg.dataset.name == 'tgif-qa' or cfg.dataset.name == 'tgif-qa-infer':
        cfg.dataset.test_question_pt = os.path.join(
                            cfg.dataset.data_dir,
                            cfg.dataset.test_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.vocab_json = os.path.join(
                            cfg.dataset.data_dir, 
                            cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(
                            cfg.dataset.video_dir, 
                            cfg.dataset.appearance_feat.format(cfg.dataset.name))

        cfg.dataset.motion_feat = os.path.join(
                            cfg.dataset.video_dir, 
                            cfg.dataset.motion_feat.format(cfg.dataset.name))
    
    
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
        cfg.dataset.motion_feat = '{}_motion_feat.h5'
        cfg.dataset.vocab_json = '{}_vocab.json'
        cfg.dataset.test_question_pt = '{}_test_questions.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids, logits = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        # print('===Question_type', cfg.dataset.question_type)
        # print('====LOGIT ', logits)
        detail = []

        if cfg.dataset.question_type in ['action', 'transition']:
            sm = torch.nn.Softmax(dim=1)
            print('origin_value::', logits.t())
            probs = sm(logits.t()) 
            print('>>>> Probs: ', type(probs), probs.size(), probs)
            detail = probs.numpy().tolist()

        elif cfg.dataset.question_type in ['frameqa']:
            sm = torch.nn.Softmax()
            probs = sm(logits) 
            print('>>>> Probs: ', type(probs), probs.size(), probs)
            answer_vocab = test_loader.vocab['answer_idx_to_token']
            values, idx = torch.topk(probs, 5)
            print('>>>Top5 ', idx)
            top_answer = []
            i = 0 
            for predict in idx:
                print('FRAMEQA-pred.item:: ', list(predict.numpy()), list(values[i].numpy()))
                # print('FRAMEQA-answer_vocab:: ', answer_vocab)
                top_answer.append(([answer_vocab[ix] for ix in list(predict.numpy())], [float(v) for v in list(values[i].numpy())] ))
                i += 1
            print('FRAMEQA-topk:: ', top_answer)
            detail = top_answer


        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")

        if cfg.dataset.question_type in ['action', 'transition']: \
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], ans_candidates[idx]]

            instances = [
                {'video_id': video_id, 
                'question_id': q_id, 
                'video_name': dict[str(q_id)][0], 
                'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                'answer': answer,
                'prediction': pred,
                'detail': d} for video_id, q_id, answer, pred, d in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds, detail)]
            # write preditions to json file
            # with open(preds_file, 'w') as f:
            #     json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            
            
            
            # Display 10 samples
            if cfg.dataset.name == 'tgif-qa-infer':
                sample_size = 1
            else: 
                sample_size = 10
                
            for idx in range(sample_size):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))

            return instances
        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]


            if cfg.dataset.question_type == 'frameqa':
                instances = [
                    {'video_id': video_id, 
                    'question_id': q_id, 
                    'video_name': str(dict[str(q_id)][0]), 
                    'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                    'answer': answer,
                    'prediction': pred,
                    'detail': d} for video_id, q_id, answer, pred, d in
                    zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds, detail)]
            else:
                instances = [
                    {'video_id': video_id, 
                    'question_id': q_id, 
                    'video_name': str(dict[str(q_id)][0]), 
                    'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                    'answer': answer,
                    'prediction': pred} for video_id, q_id, answer, pred in
                    zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            # with open(preds_file, 'w') as f:
            #     json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            
            # Display 10 samples
            if cfg.dataset.name == 'tgif-qa-infer':
                sample_size = 1
            else: 
                sample_size = 10


            for idx in range(sample_size):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))

            return instances
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='tgif_qa_action.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Config:: ', cfg)

    assert cfg.dataset.name in ['tgif-qa', 'tgif-qa-infer', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    if cfg.dataset.name != 'tgif-qa-infer':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    if cfg.dataset.name == 'tgif-qa' or cfg.dataset.name == 'tgif-qa-infer':
        cfg.dataset.test_question_pt = os.path.join(
                            cfg.dataset.data_dir,
                            cfg.dataset.test_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.vocab_json = os.path.join(
                            cfg.dataset.data_dir, 
                            cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(
                            cfg.dataset.data_dir, 
                            cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.motion_feat = os.path.join(
                            cfg.dataset.data_dir, 
                            cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
    
    
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
        cfg.dataset.motion_feat = '{}_motion_feat.h5'
        cfg.dataset.vocab_json = '{}_vocab.json'
        cfg.dataset.test_question_pt = '{}_test_questions.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids = validate(cfg, model, test_loader, device, cfg.test.write_preds)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")

        if cfg.dataset.question_type in ['action', 'transition']: \
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], ans_candidates[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': dict[str(q_id)][0], 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            
            
            
            # Display 10 samples
            if cfg.dataset.name == 'tgif-qa-infer':
                sample_size = 1
            else: 
                sample_size = 10
                
            for idx in range(sample_size):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))


        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            
            # Display 10 samples
            if cfg.dataset.name == 'tgif-qa-infer':
                sample_size = 1
            else: 
                sample_size = 10


            for idx in range(sample_size):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
