import os
import numpy as np
import time
import argparse

import yaml
from easydict import EasyDict as edict
from utils import open_file_return_det, frame_track_dict, write_tracking_result, cal_iou, get_frame_data, takefirst

parser = argparse.ArgumentParser(description='detection fusion and identity re-track post algorithm' )
parser.add_argument('--cfg', type=str, required=True, help='experiment configure file name')
parser.add_argument('--sub_task', type=str, default='MOT17', help='The dataset to be processed. e.g. MOT16>MOT17>MOT20')
parser.add_argument('--task', type=str, default='test', help='train or test')
parser.add_argument('--det_name', type=str, default='-SDP', help='detection method')
parser.add_argument('--root_path', type=str, default='./examples/data/', help='preprocess data files directory')
parser.add_argument('--img_path', type=str, default='./examples/data/', help='img files directory')
parser.add_argument('--save_path', type=str, default='./examples/res/', help='save all methods files directory')
parser.add_argument('--methods', type=str, default=['fairmot', 'gsdt', 'transtrack'], help='methods type')


def update_config(args):
    with open(args.cfg) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def run(seq, num, rect_dict_1, rect_dict_2, rect_dict_3, frame, thresh=None, thresh2=None,
        img_folder=None, det='-SDP'):  #
    for i in frame:
        print("fusing frame: " + str(i))
        dict_1 = get_frame_data(rect_dict_1, i)
        dict_2 = get_frame_data(rect_dict_2, i)
        dict_3 = get_frame_data(rect_dict_3, i)
        next_dict_1 = get_frame_data(rect_dict_1, i + 1)

        if i == num.get(seq):
            return rect_dict_1

        if len(next_dict_1) == 0:
            next_dict_2 = get_frame_data(rect_dict_2, i + 1)
            next_dict_3 = get_frame_data(rect_dict_3, i + 1)

            for k in next_dict_2:  # mattching the current frame between m2 and m3
                tt = [k[1:5] for i in next_dict_3 if cal_iou(k[1:5], i[1:5]) > 0.6]
                if len(tt) == 0:
                    continue
                f1 = -1

                for t in range(1, 11):
                    before_dict_1 = get_frame_data(rect_dict_1, i - t)

                    get_frame = [(i[0], cal_iou(tt[0], i[1:5])) for i in before_dict_1 if
                                 cal_iou(tt[0], i[1:5]) > 0.26]

                    if len(get_frame) == 0:
                        continue

                    get_frame = sorted(get_frame, key=lambda x: x[1])[0]
                    f1 = int(get_frame[0])
                    break

                if f1 == -1:
                    continue
                x1, y1, x2, y2 = tt[0]
                print("Next frame length = 0, add next frame: {}, bbox: {}".format(f1, (x1, y1, x2, y2)))
                rect_dict_1.get(i + 1).append((f1, x1, y1, x2, y2))
                rect_dict_1.get(i + 1).sort(key=takefirst)

        rect12all = [i for i in dict_1 for j in dict_2 if cal_iou(i[1:5], j[1:5]) > thresh]
        rect13all = [i for i in dict_1 for j in dict_3 if cal_iou(i[1:5], j[1:5]) > thresh]
        rect23all = [i for i in dict_2 for j in dict_3 if cal_iou(i[1:5], j[1:5]) > thresh]

        rect123all = []
        for m in rect23all:
            flag = True
            for n in dict_1:
                if cal_iou(m[1:5], n[1:5]) > thresh:
                    flag = False
            if flag:
                rect123all.append(m)

        if len(rect123all):
            vis_bbox = []
            vis_other_id = []
            vis_id = []
            for k in rect123all:
                f1 = [-1, -1]
                for t in range(1, 11):
                    before_dict_1 = get_frame_data(rect_dict_1, i - t)
                    get_frame = [(i[0], cal_iou(k[1:5], i[1:5])) for i in before_dict_1 if
                                 cal_iou(k[1:5], i[1:5]) > thresh]
                    if len(get_frame) == 0:
                        continue
                    get_frame = sorted(get_frame, key=lambda x: x[1])[0]
                    f1 = [int(get_frame[0]), i - t]
                    break

                if f1[0] == -1:
                    continue
                x1, y1, x2, y2 = k[1:5]

                if float(f1[0]) not in get_frame_data(rect_dict_1, i)[:, 0]:
                    vis_bbox.append(k[1:5])
                    vis_id.append(f1[0])
                    vis_other_id.append(k[0])
                    print("Add tracking_id : {}, bbox: {}".format(f1[0], (x1, y1, x2, y2)))

                    rect_dict_1.get(i).append((f1[0], x1, y1, x2, y2))
                    rect_dict_1.get(i).sort(key=takefirst)


def demo(opt, cfg):
    start2 = time.time()

    det = cfg[args.task.upper()][args.sub_task]
    det_name = [args.det_name]
    thresh = cfg[args.task.upper()]['THRESH'][args.sub_task]
    if args.sub_task == 'MOT20':
        det_name =['']
        opt.methods = ['fairmot', 'gsdt', 'tbw']

    for i in det:
        start1 = time.time()
        for j in det_name:
            print("Start eval: " + i + j)
            det1 = open_file_return_det(opt.methods[0], i + j,
                                        det_path=os.path.join(opt.root_path, opt.sub_task, opt.task))
            det2 = open_file_return_det(opt.methods[1], i + j,
                                        det_path=os.path.join(opt.root_path, opt.sub_task, opt.task))
            det3 = open_file_return_det(opt.methods[2], i + j,
                                        det_path=os.path.join(opt.root_path, opt.sub_task, opt.task))
            frame = det.get(i)
            frame = np.arange(1, frame+1, 1)
            det_all = [det1, det2, det3]
            # print(len(det_all[0]), len(det_all[1]), len(det_all[2]))
            lst = det_all
            rect_dict_1 = frame_track_dict(lst[0], frame)
            rect_dict_2 = frame_track_dict(lst[1], frame)
            rect_dict_3 = frame_track_dict(lst[2], frame)

            result = run(i, det, rect_dict_1, rect_dict_2, rect_dict_3, frame, thresh=thresh[i],
                         img_folder=opt.img_path)

            write_tracking_result(result, opt.sub_task, video_name=i + j, root_path=opt.save_path)

        print(i + " Finish! {:.2f}".format((time.time() - start1) / 60))

    print("ALL Finish! {:.2f}".format((time.time() - start2) / 60))


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = update_config(args)

    demo(args, cfg)
