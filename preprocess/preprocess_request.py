import preprocess_features
import preprocess_questions



def process_motion():
    video_file = '/home/kylee/work/projects/5_vqa/dataset/video/tumblr_no73q2fm0I1uuf348o1_250.gif'
    request_id = 'req1'
    question_type = 'frameqa'
    preprocess_features.preprocess_infer_motion(video_file, request_id, question_type)

def process_appearance():
    video_file = '/home/kylee/work/projects/5_vqa/dataset/video/tumblr_no73q2fm0I1uuf348o1_250.gif'
    request_id = 'req1'
    question_type = 'frameqa'
    preprocess_features.preprocess_infer_appearance(video_file, request_id, question_type)


def process_question():
    request_id = 'req1'
    question_type = 'frameqa'
    preprocess_questions.process_question(request_id, question_type)


if __name__ == '__main__':
    # process_appearance()
    # process_motion()
    process_question()