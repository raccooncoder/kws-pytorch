import json

config = dict(
    target_class = 'marvin',
    num_epochs = 20,
    batch_size = 256,
    random_seed = 13,
    img_padding_length = 130,
    enc_hidden_size = 128,
    window_size = 100,
    conv_out_channels = 16,
    conv_kernel_size = 51, 
    learning_rate = 1e-3,
    dataloader_num_workers = 8,
    weight_decay = 1e-3,
    lr_scheduler_step_size = 10,
    lr_scheduler_gamma = 0.1,
    melspec_sample_rate = 16000,
    melspec_n_mels = 64,
    melspec_n_fft = 512,
    melspec_hop_length = 128,
    melspec_f_max = 4000,
    specaug_freq_mask_param = 5,
    specaug_time_mask_param = 5,
    confidence_threshold = 0.9,
    teacher_alpha = 0.6
)

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)