import os
import random
from optparse import OptionParser
optParser = OptionParser()
optParser.add_option('-p', '--rgb_frames_path', action='store', type="string", dest='rgb_frames_path',
                     help="read rgb frames' path", default="../dataset/optical_flow_data")
option, args = optParser.parse_args()
random.seed(10)
rgb_frames_path = option.rgb_frames_path
rgb_train_test_path = "./train_val_list"
if not os.path.exists(rgb_train_test_path):
    os.makedirs(rgb_train_test_path)

label_name = ['01-angry_throw', '01-back', '01-crouch', '01-disgust_turn', '01-fear_back', '01-happy_jump',
              '01-jump',
              '01-retreat', '01-sad_squat', '01-surprise_retreat', '01-throw', '01-turn']

dir_list = os.listdir(rgb_frames_path)

item_lists = []
for root, dirs, files in os.walk(rgb_frames_path):
    for name in dirs:
        if name in label_name:
            sub_path = os.path.join(root, name)
            print(sub_path)
            if os.path.isdir(sub_path):
                sub_dir_list = os.listdir(sub_path)
                if sub_dir_list:
                    choice_frame = random.choice(sub_dir_list)
                    temp = int(choice_frame.split('_')[-1].split('.')[0])
                    if temp < 6:
                        temp = 5
                    item_list = sub_path + ' ' + str(temp) + ' ' + str(label_name.index(name)) + '\n'
                    item_lists.append(item_list)

random.shuffle(item_lists)
open(os.path.join(rgb_train_test_path, 'all_list.txt'), 'w').writelines(item_lists)
