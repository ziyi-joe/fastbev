from tqdm import tqdm
import pickle
# f = open("/root/wangqian/project/datatools/beijing_tools/output_fastbev.pkl", 'rb')
f = open("/root/ziyi/fastbev/data/waymo_test_w_label.pkl", 'rb')
tmp = pickle.load(f)
info = tmp['infos']
breakpoint()