export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/slice_sample.py --num_img_channels 1 --attention_resolutions 16 --class_cond True --diffusion_steps 1000 --dropout 0.0 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 250 --model_path logs/model100000.pt --sample_dir samples 

python scripts/slice_train.py --attention_resolutions 16 --class_cond True --diffusion_steps 1000 --dropout 0.0 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 16 --rescale_learned_sigmas True --p2_gamma 0 --p2_k 0 --log_dir train_2500 --num_img_channels 1 