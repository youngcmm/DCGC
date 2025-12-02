import os
import json

import torch
from tqdm import tqdm
import numpy as np
import gc

result_file = 'result.csv'

for dataset_name in [ "cora"]:
    gc.collect()
    torch.cuda.empty_cache()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for i in tqdm(range(1), desc="training..."):
        gc.collect()
        torch.cuda.empty_cache()
        print("\nTraining {}".format(dataset_name))
        try:
            os.system("python main.py --dataset {}".format(dataset_name))
            # print(i)
            with open('params.json', 'r') as f:
                params = json.load(f)
                print("\nTraining complete\t", end='')
                file = open(result_file, "a+")
                print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {}".format(params['acc'], params['nmi'], params['ari'],
                                                                  params['f1'], params['seed']))
                file.close()
                # file.close()
                acc_list.append(params['acc'])
                nmi_list.append(params['nmi'])
                ari_list.append(params['ari'])
                f1_list.append(params['f1'])
            f.close()
            os.remove('params.json')

        except:
            print('error')
            if os.path.exists('params.json'):
                os.remove('params.json')

        gc.collect()
        torch.cuda.empty_cache()
    # record results
    acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
    file = open(result_file, "a+")
    print('{}  {}±{}'.format(dataset_name, acc_list.mean(), acc_list.std()), file=file)
    print('{}  {}±{}'.format(dataset_name, nmi_list.mean(), nmi_list.std()), file=file)
    print('{}  {}±{}'.format(dataset_name, ari_list.mean(), ari_list.std()), file=file)
    print('{}  {}±{}'.format(dataset_name, f1_list.mean(), f1_list.std()), file=file)
    file.close()