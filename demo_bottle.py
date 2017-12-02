import os
import cv2
import torch
import numpy as np
import json
import sqlite3                                                                                                                
from torch.multiprocessing import Pool
from bottle import route, run, template, request



from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from oslo.config import cfg as cfg_oslo


def check_path_create(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
    return image, im_data, fname


# hyper-parameters
# npz_fname = 'models/yolo-voc.weights.npz'
# h5_fname = 'models/yolo-voc.weights.h5'
trained_model = cfg.trained_model
# trained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp3_158.h5')
thresh = 0.6
# ---

net = Darknet19()
net_utils.load_net(trained_model, net)
# net.load_from_npz(npz_fname)
# net_utils.save_net(h5_fname, net)
net.cuda()
net.eval()
print('load model succ...')

t_det = Timer()
t_total = Timer()


pool = Pool(processes=1)


### setting for Http server ####

# reading setting from conf
opt_morestuff_group = cfg_oslo.OptGroup(name='morestuff',
                         title='A More Complex Example')

morestuff_opts = [
    cfg_oslo.ListOpt('category', default=None,
                help=('A list of category')),
]
 
CONF = cfg_oslo.CONF
CONF.register_group(opt_morestuff_group)
CONF.register_opts(morestuff_opts, opt_morestuff_group)
CONF(default_config_files=['../webcam_img_capture/app.conf'])

det_class = CONF.morestuff.category

                                                                                                                   

conn = sqlite3.connect('../webcam_img_capture/yolo.db')

@route('/home', method='POST')
def home():
    data = request.body.read()
    body = json.loads(data)
    im_path = body['dir_path']
    #im_path = 'demo'
    im_fnames = sorted((fname for fname in os.listdir(im_path)\
                        if os.path.splitext(fname)[-1] == '.jpg'))
    im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)

    min_record_tmp_list = [0]*len(det_class)

    for i, (image, im_data, fname) in enumerate(pool.imap(preprocess, im_fnames, chunksize=1)):
        print(fname)
        t_total.tic()
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
        t_det.tic()
        bbox_pred, iou_pred, prob_pred = net(im_data)
        det_time = t_det.toc()
        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred,
                                                          prob_pred, image.shape, cfg, thresh)

        im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)


        ## create list that used to write to database
        path_list = fname.split("/")
        filename = path_list.pop()
        time_folder = im_path

        # wirte im2show to out dir
        im_out_path = os.path.join(time_folder, "out")
        check_path_create(im_out_path)
        cv2.imwrite(os.path.join(im_out_path, filename), im2show)

        tmp_list = ['0']*len(det_class)
        for i in cls_inds:
            try:
                tmp_list[det_class.index(cfg.label_names[i])] = '1'
                min_record_tmp_list[det_class.index(cfg.label_names[i])]+=1
            except:
                pass

        tmp_list.insert(0, time_folder)
        tmp_list.insert(0, filename)
        conn.execute("""insert into images_det (name, time_folder, %s)\
                        values (%s)"""%(",".join(det_class), ",".join(['?']*len(tmp_list))),
                     tmp_list)
        conn.commit()
        total_time = t_total.toc()

        if i % 1 == 0:
            format_str = 'frame: %d, (detection: %.1f Hz, %.1f ms) (total: %.1f Hz, %.1f ms)'
            print(format_str % (
                i, 1. / det_time, det_time * 1000, 1. / total_time, total_time * 1000))

            t_total.clear()
            t_det.clear()

    tmp_list = [im_path]
    min_record_tmp_list = [str(i) for i in min_record_tmp_list]
    tmp_list.extend(min_record_tmp_list)
    conn.execute("""insert into minute_det (time_folder, %s)
                    values (%s)"""%(",".join(det_class), ",".join(['?']*len(tmp_list))),
                 tmp_list)
    conn.commit()


run(host='localhost', port=5566, debug=True)

