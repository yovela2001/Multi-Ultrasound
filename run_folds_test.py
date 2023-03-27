import os


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


for i in range(5):
    os.system('python main.py --log_root BM_ViTT_img_{} --different_lr true --task BM --test_fold {} --lr 5e-6 --modality multi1 --lump 1 --epoch 40'.format(i,i))

    # os.system('python main.py --log_root BM_DeiTT_multi1_attn_0916_{} --different_lr true --task BM --test_fold {} --lr 5e-6 --modality multi1 --lump 1 --epoch 40'.format(j,j))

    # os.system('python main.py --log_root BM_DeiTT_multi1_attn2_0918_{} --different_lr true --task BM --test_fold {} --lr 1e-4 --modality multi1 --lump 1 --epoch 40'.format(j,j))
    # os.system('python main.py --log_root BM_DeiTT_multi1_attn1_0918_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality deit_attn1 --lump 1 --epoch 30'.format(i,i))

    # os.system('python main.py --log_root BM_ViTT_img_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality image --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_ViTT_video_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality video --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_resnet_img_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality image --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_resnet_video_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality video --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_ViTB_multi_1012_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_resnet_multi1_1012_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_vitt_multi1_1012_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_ViTT——multi1_1020_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_ViTT_multi1_clssim_1025_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_ViTT_multi1_clssim_1026_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(j,j))


# for i in [2]:
#     print(i)
    # os.system('python main.py --log_root BM_DeiTT_multi1_attn1_0918_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality deit_attn1 --lump 1 --epoch 30'.format(i,i))

    # os.system('python main.py --log_root BM_ViTT_img_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality image --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_ViTT_video_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality video --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_resnet_img_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality image --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_resnet_video_1007_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality video --lump 1 --epoch 30'.format(i,i))


# for i in [2,3,4]:

    # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_multi_0901_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi --lump 1 --epoch 50'.format(i,i))
    # different augmentation ColorJitter(brightness = 0.25)
    # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_multi_0902_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi --lump 1 --epoch 20'.format(i,i))

    # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_multi_importance_0902_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi_importance --lump 1 --epoch 20'.format(i,i))
    # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_multi_importance_0902_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi_importance --lump 1 --epoch 20'.format(i,i))

    # os.system('python main.py --log_root BM_ViT_multi_importance_0903_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi_importance --lump 1 --epoch 20'.format(i,i))
    # os.system('python main.py --log_root BM_ViT_image_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality image --lump 1 --epoch 20'.format(i,i))

    # os.system('python main.py --log_root BM_DeiTT_multi_importance_0904_nosoftmax_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi_importance --lump 1 --epoch 40'.format(i,i))
    # os.system('python main.py --log_root BM_DeiTT_multi1_attn_0910_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_DeiTT_multi1_noattn_0910_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_DeiTT_multi1_attn_0916_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
    # os.system('python main.py --log_root BM_DeiTT_multi1_attn2_0918_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))

    
    # os.system('python main.py --log_root BM_DeiTT_multi1_attn_0911_{} --different_lr true --task BM --test_fold {} --lr 1e-4 --modality multi1 --lump 1 --epoch 30'.format(i,i))

    # os.system('python main.py --log_root BM_DeiTB_multi_importance_0903_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi_importance --lump 1 --epoch 40'.format(i,i))
    # break
# for i in [4]:
#     print(i)
#     os.system('python main.py --log_root BM_ViTT_multi1_clssim_1025_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))

  
# for i in [2]:

#     # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_feature_similarity_0902_t10_video_{} --task BM --test_fold {} --lr 1e-5 --modality multi1 --train 0 --lump 1 --epoch 50'.format(i,i))

#     # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_img_similarity_0902_t10_video_{} --task BM --test_fold {} --lr 1e-5 --modality multi1 --train 0 --lump 1 --epoch 50'.format(i,i))

#     # 将由视频模态+图像模态取得的模型BM_ViT_multi_importance_0902用在纯视频模态的训练中
#     # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_ViT_multi_importance_0902_test_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi_importance --train 0 --lump 1 --epoch 20'.format(i,i))


#     # os.system('python main.py --log_root BM_DeiTT_multi_importance_0903_test_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --train 0 --modality multi_importance --lump 1 --epoch 50'.format(i,i))

#     # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_DeiTT_video_0903_test_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --train 0 --modality video --lump 1 --epoch 50'.format(i,i))
#     # os.system('/root/miniconda/envs/rj2/bin/python main.py --log_root BM_DeiTT_img_0903_test_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --train 0 --modality image --lump 1 --epoch 50'.format(i,i))
#     # os.system('python main.py --log_root BM_DeiTT_video_attn_0916_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality video --lump 1 --epoch 30'.format(i,i))
#     os.system('python main.py --log_root BM_DeiTT_img_attn_0916_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality image --lump 1 --epoch 30'.format(i,i))
# for i in [3]:
#     print(i)
#     os.system('python main.py --log_root BM_ViTT_multi1_clssim_1025_{} --different_lr true --task BM --test_fold {} --lr 1e-5 --modality multi1 --lump 1 --epoch 30'.format(i,i))
