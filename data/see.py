import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# in main.py
def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)


if __name__ == '__main__':
    # 加载pkl文件
    with open('./ciao_FedMF.pkl', 'rb') as f:
      [train_data, valid_data, test_data, user_id_list, item_id_list, social] = pickle.load(f)
      print("user_id_list size: ", len(user_id_list))
      print("user_id_list size: ", len(item_id_list))
      print("social size: ", len(social))
      valid_data = processing_valid_data(valid_data)
      test_data = processing_valid_data(test_data)
      print("valid data size: ", len(valid_data))
      print("test data size: ", len(test_data))
      # data: [(key, item, rate), ..]
