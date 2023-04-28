import psutil
import pynvml #导包
import time 
import os
UNIT = 1024 * 1024


pynvml.nvmlInit() #初始化
gpuDeriveInfo = pynvml.nvmlSystemGetDriverVersion()
print("Drive版本: ", str(gpuDeriveInfo, encoding='utf-8')) #显示驱动信息


gpuDeviceCount = pynvml.nvmlDeviceGetCount()#获取Nvidia GPU块数
print("GPU个数：", gpuDeviceCount )


# for i in range(gpuDeviceCount):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(i)#获取GPU i的handle，后续通过handle来处理

#     memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息

#     gpuName = str(pynvml.nvmlDeviceGetName(handle), encoding='utf-8')

#     gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)

#     gpuFanSpeed = pynvml.nvmlDeviceGetFanSpeed(handle)

#     gpuPowerState = pynvml.nvmlDeviceGetPowerState(handle)

#     gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
#     gpuMemoryRate = pynvml.nvmlDeviceGetUtilizationRates(handle).memory

#     print("第 %d 张卡："%i, "-"*30)
#     print("显卡名：", gpuName)
#     print("内存总容量：", memoryInfo.total/UNIT, "MB")
#     print("使用容量：", memoryInfo.used/UNIT, "MB")
#     print("剩余容量：", memoryInfo.free/UNIT, "MB")
#     print("显存空闲率：", memoryInfo.free/memoryInfo.total)
#     print("温度：", gpuTemperature, "摄氏度")
#     print("风扇速率：", gpuFanSpeed)
#     print("供电水平：", gpuPowerState)
#     print("gpu计算核心满速使用率：", gpuUtilRate)
#     print("gpu内存读写满速使用率：", gpuMemoryRate)
#     print("内存占用率：", memoryInfo.used/memoryInfo.total)

#     """
#     # 设置显卡工作模式
#     # 设置完显卡驱动模式后，需要重启才能生效
#     # 0 为 WDDM模式，1为TCC 模式
#     gpuMode = 0     # WDDM
#     gpuMode = 1     # TCC
#     pynvml.nvmlDeviceSetDriverModel(handle, gpuMode)
#     # 很多显卡不支持设置模式，会报错
#     # pynvml.nvml.NVMLError_NotSupported: Not Supported
#     """

#     # 对pid的gpu消耗进行统计
#     pidAllInfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)#获取所有GPU上正在运行的进程信息
#     for pidInfo in pidAllInfo:
#         pidUser = psutil.Process(pidInfo.pid).username()
#         print("进程pid：", pidInfo.pid, "用户名：", pidUser, 
#             "显存占有：", pidInfo.usedGpuMemory/UNIT, "Mb") # 统计某pid使用的显存




while True:
    flag = False 
    for i in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)#获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
        if i==0:
            continue
        if memoryInfo.free/UNIT > 23000:
            cuda_index = i 
            flag=True 
        print("第 %d 张卡："%i, "-"*30)
        # print("显卡名：", gpuName)
        print("内存总容量：", memoryInfo.total/UNIT, "MB")
        print("使用容量：", memoryInfo.used/UNIT, "MB")
        print("剩余容量：", memoryInfo.free/UNIT, "MB")

        if flag==True:
            break
            
        
    if flag==True:
        break
    time.sleep(180)
pynvml.nvmlShutdown() #最后关闭管理工具
print(cuda_index)
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_lstm_tagloss_tag'
# cuda = 'cuda:{}'.format(cuda_index)
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --use_tags_loss --encoder_lr {encoder_lr} '
        #    )
# os.system(f'python run_featurecut.py')
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# kl_training_epochs = 30
# # name = 'test_one'
# name = 'featurecut_final/transformer_featurecut30_audiocapsreal_attnweights2_epoch30_new'#ResNet38_NewTF_tagloss_keywords_1e_7
# # cuda = 'cuda:1'
# cuda = 'cuda:{}'.format(cuda_index)
# bs = 32
# decoder = 'Transformer'
# mode = 'train'
# # threshold_t = 0.4
# topk_keywords = 5
# sample_ratio = 0.8
# aug_type = 'feature'
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'featurecut_final/transformer_featurecut30_audiocapsreal_attnweights2_epoch30_new/5.pt'
# # load_pretrain_emb
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --decoder {decoder} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --dataset {dataset} --kl_training_epochs {kl_training_epochs} '
#           ) # --load_pretrain_emb 

lr = 5e-4   
encoder_lr = 5e-4
training_epochs = 30
kl_training_epochs = 30
# name = 'test_one'
name = 'featurecut_final/attmodel_clotho_base'
# cuda = 'cuda:5'
cuda = 'cuda:{}'.format(cuda_index)
bs = 32
decoder = 'AttDecoder'
sample_ratio = 0.8
aug_type = 'none'
# load_pretrain_emb
resume = 'featurecut_final/attmodel_clotho_featurecut/10.pt'
# resume = 'models_featurecut/featurecut_final/attmodel_featurecut/20.pt'
dataset = 'Clotho'
mode = 'train'
# resume ='models_featurecut/featurecut_final/attmodel_featurecut/1.pt' 
os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --mode {mode} '
          f'--aug_type {aug_type} --use_featurecut --dataset {dataset} --kl_training_epochs {kl_training_epochs} '
           )