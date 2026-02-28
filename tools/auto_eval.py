import os
import time
import subprocess
from datetime import datetime

CKPT_DIR = "/root/ziyi/product_e2e_demo-main-fastbev/fastbev/train/fastbev/work_dirs/0215"

def get_latest_ckpt(path):
    ckpts = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('epoch')]

    if not ckpts:
        return None
    ckpts.sort(key=lambda x: os.path.getmtime(x))
    return ckpts[-1]

def run_eval(ckpt_path):
    ckpt_iter = ckpt_path.split('/')[-1].split('.')[0]
    log_path = os.path.join(CKPT_DIR, f"eval_{ckpt_iter}.log")
    print(f"log save to {log_path}")
    env = os.environ.copy()
    env['LOCAL_MODE'] = 'True'

    with open(log_path, 'w') as f:
        subprocess.run(['python', 'tools/single_eval.py', 'configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v128x128x4_c192_d2_f4_train.py',
                        ckpt_path, '--eval', 'mAP'], stdout=f, stderr=subprocess.STDOUT, env=env)
    
    print(f"[]")

if __name__ == "__main__":
    last_ckpt = None
    while True:
        latest_ckpt = get_latest_ckpt(CKPT_DIR)
        if latest_ckpt and latest_ckpt != last_ckpt:
            print(f"New ckpt detectes: {latest_ckpt}")
            try:
                run_eval(latest_ckpt)
            except Exception as e:
                print(f"error: {e}")
            last_ckpt = latest_ckpt
        else:
            print("no new found")
        time.sleep(600)