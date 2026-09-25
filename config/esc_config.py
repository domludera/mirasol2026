# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The configuration for training the model
import os

exp_name = "tinyvit_3class_fold0" # the saved ckpt prefix name of the model
# Options: see model.factory.SUPPORTED_MODELS
# ("htsat", "efficientnet_b0", "fastvit", "tinyvit", "mobilenetv3", "resnet18")
model_type = "tinyvit"
workspace = "./data" # the folder of your code
dataset_path = "./data/circor" # the dataset path
desed_folder = "" # the desed file

save_preds = True

dataset_type = "esc-50" # "audioset" "esc-50" "scv2"
index_type = "full_train" # only works for audioset
balanced_data = True # class-balanced resampling of the training queue (ESC_Dataset)
dataset_npy_name = "train-circor-data-30s.npy" # binary: "train-circor-data-30s-binary.npy"

# loss_type = "clip_ce" #
# loss_type = "asl_loss"
loss_type = "weighted_clip_ce"
loss_weight = [4.0, 10.0, 1.0] # per-class weights; binary: [4.0, 1.0]

# checkpoint to resume training from, or to evaluate with `main_moody.py test`
resume_checkpoint = None # e.g. "data/results/<exp>/checkpoint/lightning_logs/version_0/checkpoints/<ckpt>.ckpt"

esc_fold = 0 # just for esc dataset, select the fold you need for evaluation and (+1) validation

debug = False

random_seed = 123777 # 19970318 970131 12412 127777 1009* 34047

# for signal processing
sample_rate = 4000  
clip_seconds = 30
clip_samples = sample_rate * clip_seconds
window_size = 128   
hop_size = 120       # 120,000 / 1024 frames = ~117 frames (will be padded/interpolated to 1024)
mel_bins = 64       
fmin = 25           
fmax = 2000         

shift_max = int(clip_samples * 0.5)

batch_size = 16 
learning_rate = 5e-4 
max_epoch = 150
num_workers = 16 
save_top_k = 2
val_check_interval = 1

lr_scheduler_epoch = [10,20,30]
lr_rate = [0.05, 0.1, 0.2]

enable_time_shift = True 
enable_repeat_mode = False 
enable_tscam = True 

# for data collection
classes_num = 3 # 3-class: Present (0) / Unknown (1) / Absent (2); binary: Present (0) / Absent (1)
# patch_size = (25, 4) # deprecated
# crop_size = None # int(clip_samples * 0.5) deprecated

# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size = 256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96 
htsat_depth = [2,2,6,2]

swin_pretrain_path = None
# "/home/Research/model_backup/pretrain/swin_tiny_c24_patch4_window8_256.pth"

# Some Deprecated Optimization in the model design, check the model code for details
htsat_attn_heatmap = False
htsat_hier_output = False 
htsat_use_max = False


# for ensemble test 

ensemble_checkpoints = []
ensemble_strides = []


# weight average folder
wa_folder = os.path.join(workspace, "results", exp_name, "checkpoint", "lightning_logs", "version_17", "checkpoints")
# weight average output filename
wa_model_path = "l-epoch=89-acc=0.917.ckpt"
# wa_model_path = None

fl_local = False # indicate if we need to use this dataset for the framewise detection
