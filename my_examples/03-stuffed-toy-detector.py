import cv2

from fastbook import *
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from torchvision.models.quantization import resnet18

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

path = Path('stuffed_toys')

def learn_from_images():
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(92, method='squish')],
        batch_tfms = aug_transforms
    ).dataloaders(path)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    return learn


def analyze_learner(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(5, nrows=1, figsize=(17,4))


def clean_data(learn):
    from fastai.vision.widgets import ImageClassifierCleaner
    cleaner = ImageClassifierCleaner(learn)
    for idx in cleaner.delete():
        try:
            cleaner.fns[idx].unlink()
        except:
            pass
    for idx, cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path / cat)


def add_name_to_image(character, frame):
    if not character or character == '':
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 200)
    font_scale = 6
    color = (255, 0, 0)
    thickness = 25

    frame = cv2.putText(frame, character, org, font,
                        font_scale, color, thickness, cv2.LINE_AA)
    return frame


def get_video_and_process_frame(learner):
    cv2.startWindowThread()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print('Cannot open camera')
        exit()
    
    old_time = time.time_ns()
    capture_time_in_s = 0.5 * 1000 * 1000 * 1000
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        new_time = time.time_ns()
        if new_time - old_time > capture_time_in_s:
            old_time = new_time
            ret, frame = cap.read()
        
            if not ret:
                print('Cannot read from camera')
                break
            
            frame = detect_person(learner, frame)
            cv2.namedWindow("frame")
            cv2.imshow('frame', frame)
    
    cv2.waitKey(1)
    cap.release()
    del cap
    cv2.waitKey(1)
    cv2.destroyWindow('frame')
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def detect_person(learn, frame):
    pred = learn.predict(frame)
    print(pred)
    pred = pred[0]
    frame = add_name_to_image(pred, frame)
    return frame

# def split_video():
#     capture = cv2.VideoCapture('/Users/lukaswoodtli/Development/fastai-course/my_examples/stuffed_toys/Beran/IMG_8814.MOV')
#
#     frameNr = 0
#
#     while (True):
#
#         success, frame = capture.read()
#
#         if success:
#             cv2.imwrite(f'/Users/lukaswoodtli/Development/fastai-course/my_examples/stuffed_toys/Beran/frame_{frameNr}.jpg', frame)
#
#         else:
#             break
#
#         frameNr = frameNr + 1
#
#     capture.release()

def main():
    # Training
    learner = learn_from_images()
    analyze_learner(learner)
    # clean_data(learn)

    # Inference
    get_video_and_process_frame(learner)


if __name__ == '__main__':
    main()
    #split_video()
