import os


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


for i in range(5):
    os.system('python main.py --log_root BM_DeiT_img_test_{} --different_lr true --task BM --test_fold {} --train 0 --modality multi1 --lump 1'.format(i,i))

 