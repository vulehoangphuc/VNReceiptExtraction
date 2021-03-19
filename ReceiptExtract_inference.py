#----------------------DETECTION------------------------------------------------
import sys
sys.path.append('./CRAFTpytorch/')
import os
import time
import argparse
import json
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import pandas as pd
import CRAFTpytorch.test as test

from CRAFTpytorch.test import copyStateDict
from craft import CRAFT

from collections import OrderedDict

from google.colab.patches import cv2_imshow

def load_detection_model():
  parser = argparse.ArgumentParser(description='CRAFT Text Detection')
  parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
  parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
  parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
  parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
  parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
  parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
  parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
  parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
  parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
  parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
  parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
  parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
  args = parser.parse_args(["--trained_model=./models/craft_mlt_25k.pth","--refine", "--refiner_model=./models/craft_refiner_CTW1500.pth"])
  net = CRAFT()     # initialize
  print('Loading weights from checkpoint (' + args.trained_model + ')')
  if args.cuda:
      net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
  else:
      net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

  if args.cuda:
      net = net.cuda()
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = False

  net.eval()

  # LinkRefiner
  refine_net = None
  if args.refine:
      from refinenet import RefineNet
      refine_net = RefineNet()
      print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
      if args.cuda:
          refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
          refine_net = refine_net.cuda()
          refine_net = torch.nn.DataParallel(refine_net)
      else:
          refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

      refine_net.eval()
      # args.poly = True
  return net,refine_net,args

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
def infer_detection(impath,net,refine_net,args):
  #CRAFT
  """ For test images in a folder """
  image_list, _, _ = file_utils.get_files(impath)

  image_paths = []
  image_names = []
  #CUSTOMISE START
  start = impath

  result_folder = './Results/'
  data={}
  
  t = time.time()

  # load data
  for k, image_path in enumerate(image_list):
    print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)

    image_name=os.path.relpath(image_path, start)

    bboxes, polys, score_text, det_scores = test.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)
    bbox_score={}
    index=0
    for box,conf in zip(bboxes,det_scores):
      bbox_score[str(index)]={}
      bbox_score[str(index)]['detconf']=str(conf)
      bbox_score[str(index)]['box']=[]
      for coors in box:
        temp=[str(coors[0]),str(coors[1])]
        bbox_score[str(index)]['box'].append(temp)
      index+=1
    data[image_name]=bbox_score

    # for box_num in range(len(bboxes)):
    #   key = str (det_scores[box_num])
    #   item = bboxes[box_num]
    #   bbox_score[key]=item

    # data['word_bboxes'][k]=bbox_score

    # save score text
    # filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)

    # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

  if not os.path.isdir('./Results'):
    os.mkdir('./Results')
  # data.to_csv('./Results_csv/data.csv', sep = ',', na_rep='Unknown')
  # print(data)
  with open('./Results/data.json', 'w') as jsonfile:
    json.dump(data, jsonfile)
    jsonfile.close()
  print("elapsed time : {}s".format(time.time() - t))

# imgtest=io.imread("/content/Image/19414.jpg")
# imgtest=np.array(imgtest)

# imgtest='/content/Image'
# craft_net,craft_refine_net,craft_args=load_detection_model()
# infer_detection(imgtest,craft_net,craft_refine_net,craft_args)

#-----------------RECOGNITION---------------------------------------------------
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
def load_recognition_model():
  #chuan bi ocr predict model
  config = Cfg.load_config_from_file('./vietocr/config.yml')
  config['weights'] = "./models/transformerocr.pth"
  config['cnn']['pretrained']=False
  config['device'] = 'cuda:0'
  config['predictor']['beamsearch']=False
  recognizer = Predictor(config)
  return recognizer

def crop(pts, image):

  """
  Takes inputs as 8 points
  and Returns cropped, masked image with a white background
  """
  for i in pts:
    if (i[0]<0):
      i[0]=0
    if(i[1]<0):
      i[1]=0
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  #print('x,y,w,h:',x,y,w,h)
  #print(image)
  cropped = image[y:y+h, x:x+w].copy()
  pts = pts - pts.min(axis=0)
  mask = np.zeros(cropped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(cropped, cropped, mask=mask)
  bg = np.ones_like(cropped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg + dst

  return dst2
def generate_words(image_name, score_bbox, image,recognizer,tsvdata,pick_path):

  #score_bbox: {'0': {'detconf': '0.886273', 'box': [['604.8', '116.8'], ['737.6', '116.8'], ['737.6', '209.6'],...
  num_bboxes = len(score_bbox)
  data=open(pick_path + image_name.replace('.jpg','.tsv'),'w')

  for num in range(num_bboxes): #duyet qua moi bbox trong 1 image
    bbox_coords = score_bbox[str(num)]['box']
    if(bbox_coords):
      l_t = float(bbox_coords[0][0])
      t_l = float(bbox_coords[0][1])
      r_t = float(bbox_coords[1][0])
      t_r = float(bbox_coords[1][1])
      r_b = float(bbox_coords[2][0])
      b_r = float(bbox_coords[2][1])
      l_b = float(bbox_coords[3][0])
      b_l = float(bbox_coords[3][1])
      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
      #print('pts:',pts)
      if np.all(pts) > 0:
        # break
        word = crop(pts, image)
        img=Image.fromarray(word)
        trans=recognizer.predict(img)
        coords=tsvdata[image_name][str(num)]['box']
        coords=["1",coords[0][0],coords[0][1],coords[1][0],coords[1][1],coords[2][0],coords[2][1],coords[3][0],coords[3][1]]
        tsvdata[image_name][str(num)]['trans']=trans
        coords.append(trans)
        coords=",".join(coords)
        data.write(coords)
        data.write('\n')
        # folder = '/'.join( image_name.split('/')[:-1])
        # folder=image_name


        # #CHANGE DIR
        # if not os.path.isdir('./cropped_words'):
        #   os.mkdir('./cropped_words')
        # dir = './cropped_words/'

        # if not os.path.isdir(os.path.join(dir + folder)):
        #   os.mkdir(os.path.join(dir + folder))
        # dir=dir+folder+'/'

        # try:
        #   # print(image_name)
        #   file_name = os.path.join(dir + image_name+'_'+str(num))
        #   # cv2.imwrite(file_name+'_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l), word)
        #   cv2.imwrite(file_name+'.jpg',word)
        #   #print('Image saved to '+file_name+'_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l))
        # except:
        #   continue
  data.close()


def crop_OCR(recognizer):
  data=json.load(open('./Results/data.json')) #PATH TO detected texts results

  start = './temp' #PATH TO INPUT TEST IMAGES
  pick_path='./txt_format/' #OUTPUT TXT FILES PATH 
  if not os.path.isdir(pick_path):
    os.mkdir(pick_path)

  for image_name in data:
    image = cv2.imread(os.path.join(start,image_name))
    score_bbox = data[image_name]
    # image_name = image_name.strip('.jpg')# strip doesn't mean "remove this substring". x.strip(y)
    # treats y as a set of characters and strips any characters in that set from both ends of x
    generate_words(image_name, score_bbox, image,recognizer,data,pick_path)
  with open('./Results/data.json', 'w') as jsonfile:
    json.dump(data, jsonfile)
    jsonfile.close()

#-----------------KEY EXTRACTION------------------------------------------------

import sys
sys.path.append('./PICKpytorch/')
from tqdm import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
def load_extraction_model():
  args = argparse.ArgumentParser(description='PyTorch PICK Testing')
  args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                    help='path to load checkpoint (default: None)')
  args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                    help='ocr results folder including boxes and transcripts (default: None)')
  args.add_argument('--impt', '--images_path', default=None, type=str,
                    help='images folder path (default: None)')
  args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                    help='output folder (default: predict_results)')
  args.add_argument('-g', '--gpu', default=-1, type=int,
                    help='GPU id to use. (default: -1, cpu)')
  args.add_argument('--bs', '--batch_size', default=1, type=int,
                    help='batch size (default: 1)')
  # args = args.parse_args(["--checkpoint=./PICKpytorch/model/model_best.pth","--boxes_transcripts="+test_box_path,"--images_path="+test_img_path,"--output_folder=./extraction_results/","--gpu=0","--batch_size=2"])
  args = args.parse_args(["--checkpoint=./models/PICK_model.pth","--output_folder=./extraction_results/","--gpu=0","--batch_size=2"])
  device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
  checkpoint = torch.load(args.checkpoint, map_location=device)

  config = checkpoint['config']
  state_dict = checkpoint['state_dict']
  monitor_best = checkpoint['monitor_best']
  print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

  # prepare model for testing
  pick_model = config.init_obj('model_arch', pick_arch_module)
  pick_model = pick_model.to(device)
  pick_model.load_state_dict(state_dict)
  pick_model.eval()
  return pick_model,args,device

def extractKeys(test_box_path,test_img_path,pick_model,args,device):
  # setup dataset and data_loader instances
  test_dataset = PICKDataset(boxes_and_transcripts_folder=test_box_path,
                              images_folder=test_img_path,
                              resized_image_size=(480, 960),
                              ignore_error=False,
                              training=False)
  test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=2, collate_fn=BatchCollateFn(training=False))

  # setup output path
  output_path = Path(args.output_folder)
  output_path.mkdir(parents=True, exist_ok=True)

  # predict and save to file
  with torch.no_grad():
      for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
          for key, input_value in input_data_item.items():
              if input_value is not None:
                  input_data_item[key] = input_value.to(device)

          output = pick_model(**input_data_item)
          logits = output['logits']
          new_mask = output['new_mask']
          image_indexs = input_data_item['image_indexs']  # (B,)
          text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
          mask = input_data_item['mask']
          # List[(List[int], torch.Tensor)]
          best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
          predicted_tags = []
          for path, score in best_paths:
              predicted_tags.append(path)

          # convert iob index to iob string
          decoded_tags_list = iob_index_to_str(predicted_tags)
          # union text as a sequence and convert index to string
          decoded_texts_list = text_index_to_str(text_segments, mask)

          for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
              # List[ Tuple[str, Tuple[int, int]] ]
              spans = bio_tags_to_spans(decoded_tags, [])
              spans = sorted(spans, key=lambda x: x[1][0])

              entities = []  # exists one to many case
              for entity_name, range_tuple in spans:
                  entity = dict(entity_name=entity_name,
                                text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                  entities.append(entity)

              result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
              ent={}
              for item in entities:
                ent.setdefault(item['entity_name'],[]).append((item['text']))
              for k,v in ent.items():
                ent[k]=' '.join(i for i in ent[k])
              with open(result_file,'w') as fw:
                for k,v in ent.items():
                  fw.write(k+":   "+v)
                  fw.write("\n")
                fw.close()

# UI

import gradio as gr
import numpy as np
import os
from PIL import Image
import sys
import matplotlib.pyplot as plt
def sepia(img):
  temp='./temp/'
  if not os.path.isdir(temp):
    os.mkdir(temp)
  img = Image.fromarray(img)
  imgpath=temp+"image.jpg"
  img.save(imgpath)

  print('Detecting texts...')
  
  infer_detection(temp,craft_net,craft_refine_net,craft_args)

  print('Completed!','\n','Cropping and Recognizing...')
  
  crop_OCR(recognizer)

  print('Completed!','\n','Extracting keys...')
  extractKeys('./txt_format/',temp,pick_model,pick_args,device)
  print('Completed!')
  re=open('./extraction_results/'+'image.txt','r').read()
  return re

sample_images = [
                 ["19500.jpg"],["20470.jpg"],
                 ["20736.jpg"],["21337.jpg"],
                 ["19473.jpg"],["21829.jpg"],
                 ["21813.jpg"],
                 ["21803.jpg"],["21792.jpg"],
                 ["21839.jpg"],["21845.jpg"],
                 ["21868.jpg"]
]
sample_images = [["./demo/"+i[0]] for i in sample_images]
craft_net,craft_refine_net,craft_args=load_detection_model()
recognizer=load_recognition_model()
pick_model,pick_args,device=load_extraction_model()
iface = gr.Interface(fn=sepia,inputs=gr.inputs.Image(),outputs="text",examples=sample_images)
# iface = gr.Interface(fn=sepia,inputs=gr.inputs.Image(),outputs="text")
iface.launch(debug=False,share=True)
input("Demo is Running.... (type something to stop!)")
shutil.rmtree("./Results")
shutil.rmtree("./txt_format")
shutil.rmtree("./temp")
# iface.launch(debug=False)
# iface.launch(debug=True)

# https://www.gradio.app/ml_examples
