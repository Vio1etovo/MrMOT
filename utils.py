import os
import numpy as np
import cv2


def cal_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    i_w = min(x2, x4) - max(x1, x3)
    i_h = min(y2, y4) - max(y1, y3)
    if (i_w <= 0 or i_h <= 0):
        return 0
    i_s = i_w * i_h
    s_1 = (x2 - x1) * (y2 - y1)
    s_2 = (x4 - x3) * (y4 - y3)
    return float(i_s) / (s_1 + s_2 - i_s)


def open_file_return_det(dn, video_name, det_path=' '):
    open_file = open(os.path.join(det_path, dn + '-' + video_name + '.txt'), 'r+')
    det = [i.split(',')[0:6] for i in open_file]
    return det


def frame_track_dict(det, frame):
    d = {}
    dd = d.fromkeys(frame)
    for key in dd:
        dd[key] = []
    for i in det:
        tracking_id, x1, y1, w, h, = i[1:6]
        x1, y1, x2, y2 = float(x1), float(y1), float(x1) + float(w), float(y1) + float(h)
        dd[int(i[0])].append((int(tracking_id), x1, y1, x2, y2))  # append里如果有一个是字符其余都会被变成字符

    return dd


def frame_dict_subtrackid(det, frame, num):
    d = {}
    dd = d.fromkeys(frame)
    for key in dd:
        dd[key] = []
    for i in det:
        tracking_id, x1, y1, w, h, = i[1:6]
        x1, y1, x2, y2 = float(x1), float(y1), float(x1) + float(w), float(y1) + float(h)
        dd[int(i[0])].append((int(tracking_id) - num, x1, y1, x2, y2))  # append里如果有一个是字符其余都会被变成字符
    return dd


def frame_dict_subframe(det, frame, num):
    d = {}
    dd = d.fromkeys(frame)
    for key in dd:
        dd[key] = []
    for i in det:
        tracking_id, x1, y1, w, h, = i[1:6]
        x1, y1, x2, y2 = float(x1), float(y1), float(x1) + float(w), float(y1) + float(h)
        dd[int(i[0])-num].append((int(tracking_id), x1, y1, x2, y2))  # append里如果有一个是字符其余都会被变成字符
    return dd


def get_frame_data(dict, frame):
    data = []
    if dict.get(frame) != None and dict.get(frame) != []:  # 如果下一帧本来就没有检测出数据那就是空集
        data = np.array(dict.get(frame))[:, 0:5]
    return data


def next_mattch(data1, data2):
    next_id, next_bbox = None, None
    for k in data1:
        if k[0] != data2[0]:
            continue
        else:
            next_id, next_bbox = k[0], k[1:5]

    return next_id, next_bbox


def takefirst(elem):
    return elem[0]


def takesecond(elem):
    return elem[1]


def visualization(model_name, seq, frame, bboxes, tracking_id=None, img_folder=None, output_dir=None):
    dir_path = os.path.join(img_folder, seq, 'img1')
    img_path = [(x.split('.')[0], os.path.join(dir_path, x)) for x in os.listdir(dir_path)]
    img = [i for i in img_path if frame == int(i[0])]
    img_frame = int(img[0][0])
    img_path = img[0][1]
    img = cv2.imread(img_path)

    fairmot = {'MOT17-01-SDP': 4744, 'MOT17-03-SDP': 4880, 'MOT17-06-SDP': 5375, 'MOT17-07-SDP': 5832,
               'MOT17-08-SDP': 6076, 'MOT17-12-SDP': 6279, 'MOT17-14-SDP': 6479}
    gsdt = {'MOT17-01-SDP': 0, 'MOT17-03-SDP': 77, 'MOT17-06-SDP': 625, 'MOT17-07-SDP': 1023,
            'MOT17-08-SDP': 1197, 'MOT17-12-SDP': 1372, 'MOT17-14-SDP': 1535}

    if tracking_id != None:
        for i in range(len(bboxes)):
            bboxes[i][2] = bboxes[i][2] - bboxes[i][0]
            bboxes[i][3] = bboxes[i][3] - bboxes[i][1]

        # tracking_id = [i-fairmot[seq] for i in tracking_id]
        online_im = vis.plot_tracking(img, bboxes, tracking_id, frame_id=img_frame, fps=0)
        if not os.path.exists(output_dir + seq):
            os.mkdir(output_dir + seq)
        cv2.imwrite(os.path.join(output_dir, seq, model_name + '-{:05d}.jpg'.format(frame)), online_im)
    else:
        tracking_tlwhs = []
        tracking_id = []
        for i in bboxes:
            id, bbox = int(i[0]), i[1:5]
            id = id - fairmot[seq]
            x1, y1, x2, y2 = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))
            bbox = x1, y1, x2-x1, y2-y1
            bbox = np.array(bbox)
            tracking_tlwhs.append(bbox)
            tracking_id.append(id)
        online_im = vis.plot_tracking(img, tracking_tlwhs, tracking_id, frame_id=frame, fps=0)
        if not os.path.exists(output_dir + seq):
            os.mkdir(output_dir + seq)
        cv2.imwrite(os.path.join(output_dir, seq, model_name + '-{:05d}.jpg'.format(frame)), online_im)


def merge_rect(rect_dict1, rect_dict2, frame):
    for k in frame:
        bboxes1 = rect_dict1.get(k)
        bboxes2 = rect_dict2.get(k)
        for i in range(len(bboxes1)):
            bbox1 = bboxes1[i][0:4]
            for j in range(len(bboxes2)):
                bbox2 = bboxes2[j][0:4]
                smi = cal_iou(bbox1, bbox2)
                if smi > 0.6:
                    break
                elif j == len(bboxes2) - 1:
                    rect_dict2[k].append(bboxes1[i])

    return rect_dict2


def write_result(result, file_name, det_path='/home/zbx/Documents/post/Three/'):
    file_trades = open(os.path.join(det_path, file_name + 'MOT17-04.txt'), 'a+')
    for i in result:
        id0 = 0
        bboxes = result[i]
        for j in bboxes:
            id0 += 1
            x1, y1, x2, y2, conf = j
            file_trades.write(str(i) + ',' + str(id0) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ',' + str(
                y2 - y1) + ',' + str(conf))

    file_trades.close()


def write_tracking_result(result, filename, root_path='/home/zbx/Documents/post/Three/MOT17/val/exp4/add/',
                          video_name='MOT17-04-FRCNN'):
    file = open(os.path.join(root_path, filename + video_name + '.txt'), 'a+')
    for i in result:
        bboxes = result[i]
        for j in bboxes:
            tracking_id, x1, y1, x2, y2 = j
            file.write(str(i) + ',' + str(int(tracking_id)) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ','
                       + str(y2 - y1) + ',' + str(1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')

    file.close()















