
import os
import json
import numpy as np

class JSON_Manager:
    """
    This class is used in order to be able to resume the execution of an interrupted training
    """ 
    def __init__(self, author, model_name):
        self.model_name = model_name
        self.author = author
        self.path = os.path.join("json_logs", author, model_name+".json")
        
        try:
            os.mkdir(os.path.join("json_logs", author))
        except FileExistsError:
            pass

        try:
            with open(self.path) as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {'epochs': 0,
                    'patience_cnt': 0,
                    'best_metric': 99999,
                    'best_conf_mat':None,
                    'best_epoch': -1}

        self.data = data
        print(data)
     
    def _save(self):
        with open(self.path, 'w+') as outfile:
            json.dump(self.data, outfile)
        return

    def get_epoch(self):
        return self.data['epochs']

    def increase_epoch(self):
        self.data['epochs'] += 1
        self._save()

    def get_patience_cnt(self):
        return self.data['patience_cnt']

    def increase_patience_cnt(self):
        self.data['patience_cnt'] += 1
        self._save()
    
    def reset_patience_cnt(self):
        self.data['patience_cnt'] = 0
        self._save()

    def get_best_metric(self):
        return self.data['best_metric']

    def set_best_metric(self, value):
        self.data['best_metric'] = value
        self._save()

    def get_best_conf_mat(self):
        return np.array(self.data['best_conf_mat'])

    def set_best_conf_mat(self, mat):
        self.data['best_conf_mat'] = mat.tolist()
        self._save()

    def get_best_epoch(self):
        return self.data['best_epoch']

    def set_best_epoch(self, ep):
        self.data['best_epoch'] = ep
        self._save()