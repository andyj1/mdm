# export TORCH_CUDNN_V8_API_DISABLED=1
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=$1

python distill_final.py \
    --name $2 \
    --dataset flickr8k \
    --buffer_path ./buffer/flickr8k/nfnet_bert/InfoNCE \
    --num_queries 500 \
    --batch_size_train 256 \
    --batch_size_test 256 \
    --lr_txt 1000 \
    --lr_img 1000 \
    --teacher_resample 1 \
    --Iteration 3000 \
    --epoch_eval_train 100 \
    --w_nce 1.0 \
    --w_sph_u_mmd 0.8 \
    --sph_u_mmd_sigma 0.5 \
    --w_sph_g_mmd 0.8 \
    --sph_g_mmd_sigma 0.5 \
    --c_sigma_x 0.5 \
    --c_sigma_y 0.5 \
    --c_sigma_u 0.2 \
    --syn_init kmeans \
    --init_model_method mixed \
    --w_cross_cov 0.0 \
    --merge_alpha 0.5 \
    --min_start_epoch 1 \
    --max_start_epoch 10 \
    --diversity_weight_u 1 \
    --diversity_weight_g 1 \
    --diversity_weight_ortho 1 \
    --sqrtmmd True \
    --logmmd False \
    --log_dir logs \
    --eval_eval_freq 100 \
    --eval_it 50 \
    --save_it 50 \
    --log_freq 1 \
    --num_eval 5 \
    --wandb \
    --wandb_entity your_username \
    --wandb_project your_project_name \
    --wandb_mode online \
    --kmeans_viz False \
    --wall_clock_tracker \
    --seed 1 \