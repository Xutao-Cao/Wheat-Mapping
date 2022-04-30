import os
import datetime
import numpy as np
from datetime import date

def make_data_set_1(site_range, pid):

  IMG_HEIGHT = 256
  IMG_WIDTH = 256
  years = [2017, 2018, 2019, 2020]
  # site_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  base_path = os.path.join("D:\\","Graduation_Thesis_Dataset")
  S1_path = os.path.join(base_path, "S1_dataset")
  L8_path = os.path.join(base_path, "L8_dataset")
  All_features_path = os.path.join(base_path, "All_features")
  SAR_features_path = os.path.join(base_path, "SAR_features")
  Sp_features_path = os.path.join(base_path, "Sp_features")
  CDL_path = os.path.join(base_path, "CDL_label")
  # for satellite in satellites:
  #   cur_path = None
  #   if satellite == 'Landsat_8':
  #     cur_path = L8_path
  #   elif satellite == 'S1':
  #     cur_path = S1_path
  for site in site_range:
    # print(cur_path)
    cur_path = None
    site_path = str(site)
    # print(cur_path)
    for year in years:
      year_path = os.path.join(site_path,str(year))
      cdlpath = "CDL_" + str(site) + '_' + str(year) + '.npy'
      cdlpath = os.path.join(year_path, cdlpath)
      cdlpath = os.path.join(L8_path, cdlpath)
      label = np.load(cdlpath)
      start_date = date(year, 9, 10)
      SAR_list = []
      SP_list = []
      for advancement in range(1, 11):
        cur_time = start_date + datetime.timedelta(days=advancement*24)
        cur_time_str = cur_time.strftime("%Y%m%d")
        S1_cur_image_path = os.path.join(year_path, 'S1_' + str(site) + '_'+ cur_time_str + '.npy')
        L8_cur_image_path = os.path.join(year_path, 'Landsat_8_' + str(site) + '_'+ cur_time_str + '.npy')
        S1_img_path = os.path.join(S1_path, S1_cur_image_path)
        L8_img_path = os.path.join(L8_path, L8_cur_image_path)
        SP_list.append(np.load(L8_img_path))
        SAR_list.append(np.load(S1_img_path))
      SAR_img = np.stack(SAR_list, axis=-1)
      SP_img = np.stack(SP_list, axis=-1)
      All_img = np.concatenate((SP_img, SAR_img), axis=2)
      img_cnt = 0
      x, y, c, t = All_img.shape
      x_steps = x//IMG_WIDTH
      y_steps = y//IMG_HEIGHT
      for i in range(x_steps):
        for j in range(y_steps):
          break_flg = 0
          for w in range(i*IMG_WIDTH, i*IMG_WIDTH + IMG_WIDTH//2):
            if break_flg or w > x - 1:
              break
            for h in range(j*IMG_HEIGHT, j*IMG_HEIGHT + IMG_HEIGHT//2):
              if h > y-1:
                break
              if not np.any(np.isnan(All_img[w : w+IMG_WIDTH, h : h+IMG_HEIGHT, :, :])) and not np.any(np.isnan(label[w : w+IMG_WIDTH, h : h+IMG_HEIGHT,])):
                all_fea_path = os.path.join(All_features_path, str(site) + '_'+ str(year) +"_" +str(img_cnt) + '.npy')
                label_path = os.path.join(CDL_path, str(site) + '_'+ str(year) +"_"+ str(img_cnt) + '.npy')
                SAR_img_path = os.path.join(SAR_features_path, str(site) + '_'+ str(year) +"_"+ str(img_cnt) + '.npy')
                Sp_img_path = os.path.join(Sp_features_path, str(site) + '_'+ str(year) +"_"+ str(img_cnt) + '.npy')
                np.save(all_fea_path, All_img[w : w+IMG_WIDTH, h : h+IMG_HEIGHT, :, :])
                np.save(SAR_img_path, SAR_img[w : w+IMG_WIDTH, h : h+IMG_HEIGHT, :, :])
                np.save(Sp_img_path, SP_img[w : w+IMG_WIDTH, h : h+IMG_HEIGHT, :, :])
                np.save(label_path, label[w : w+IMG_WIDTH, h : h+IMG_HEIGHT])
                break_flg = True
                img_cnt += 1  
                file_path = os.path.join(CDL_path, str(site) + '_'+ str(year) + '.txt')
                f = open(file_path, "a")
                f.write(str(img_cnt)+": [ "+str(w)+", "+str(h) + " ]   [ " + str(w + IMG_WIDTH -1) + ", " + str(h + IMG_HEIGHT -1) + " ]\n")
                f.close()
                break
      print(str(site)," ",  str(year), ": ", str(img_cnt), "images, ", img_cnt/(x_steps*y_steps))
  return pid