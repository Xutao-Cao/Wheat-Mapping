import torch
import os
import datetime
import sklearn.metrics as me
import numpy as np
import csv
class MultiTrainHelper():
    def __init__(self, net) -> None:
        pass

class train_mission():
    def __init__(self, train_site, test_site, type, model_name, idx, result_path_base = './Logs',test_years = [2020,] ,train_years = [2017, 2018, 2019],) :
        self.index = idx
        self.model_name = model_name
        self.train_years = train_years
        self.test_years = test_years
        self.train_site = train_site
        self.test_site = test_site
        self.begin_time = None
        self.end_time = None
        self.type = type
        self.real_path = os.path.join(result_path_base, model_name+'.csv')
        self.scores = {}

    def mission_start(self):
        self.begin_time = datetime.datetime.now()
        begintime_fmt = self.begin_time.strftime("%Y-%m-%d %H:%M:%S")
        print(begintime_fmt, "Index:{i}, Model: {m}, train_site: {s1}, test_site: {s2}, type: {t}".format(m = self.model_name, s1 = self.train_site[0], t=self.type, i=self.index, s2 = self.test_site[0]))

    def mission_get_score(self, y_train, y_train_predict, y_test, y_test_predict):
        self.end_time = datetime.datetime.now()
        write_list = []
        write_list.append(self.begin_time.strftime("%Y-%m-%d %H:%M:%S"))
        write_list.append(self.end_time.strftime("%Y-%m-%d %H:%M:%S"))
        write_list.append(self.train_site[0])
        write_list.append(self.test_site[0])
        write_list.append(self.type)
        acc_train = me.accuracy_score(y_train, y_train_predict)
        acc_test = me.accuracy_score(y_test, y_test_predict)
        f1_train = me.f1_score(y_train, y_train_predict)
        f1_test = me.f1_score(y_test, y_test_predict)
        kappa_train = me.cohen_kappa_score(y_train, y_train_predict)
        kappa_test = me.cohen_kappa_score(y_test, y_test_predict)
        write_list.append(acc_train)
        write_list.append(acc_test)
        write_list.append(f1_train)
        write_list.append(f1_test)
        write_list.append(kappa_train)
        write_list.append(kappa_test)
        # np.save("./Trained models/{m}_{t}_{s1}_{s2}_y_train.npy".format(m = self.model_name, t = self.type, s1 = self.train_site[0], s2 = self.test_site[0]), y_train)
        # np.save("./Trained models/{m}_{t}_{s1}_{s2}_y_test.npy".format(m = self.model_name, t = self.type, s1 = self.train_site[0], s2 = self.test_site[0]), y_test)
        # np.save("./Trained models/{m}_{t}_{s1}_{s2}_y_train_predict.npy".format(m = self.model_name, t = self.type, s1 = self.train_site[0], s2 = self.test_site[0]), y_train_predict)
        # np.save("./Trained models/{m}_{t}_{s1}_{s2}_y_test_predict.npy".format(m = self.model_name, t = self.type, s1 = self.train_site[0], s2 = self.test_site[0]), y_test_predict)
        time_elapsed = self.end_time - self.begin_time
        time_elapsed = time_elapsed / 60
        write_list.append(time_elapsed) 
        with open(self.real_path,"a+",) as f:
            csv_file = csv.writer(f, )
            csv_file.writerow(write_list)
    
    def save_model(self, net):
        torch.save(net.state_dict(), "./Trained models/{m}_{t}_{s1}_{s2}.pth".format(m = self.model_name, t = self.type, s1 = self.train_site[0], s2 = self.test_site[0]))
