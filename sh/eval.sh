DATASET=flickr8k
DEVICE=0
NUM_QUERIES=100
IMAGE_ENCODER=nfnet
TEXT_ENCODER=bert
NAME=nfnet_bert
NUM_EVAL=5
EPOCH_EVAL_TRAIN=100
EVAL_EVAL_FREQ=100
BATCH_SIZE_TRAIN=1024
BATCH_SIZE_TEST=1024

CUDA_VISIBLE_DEVICES=$DEVICE python distill_final_crossarch_8k100.py \
    --num_queries $NUM_QUERIES \
    --num_eval $NUM_EVAL \
    --epoch_eval_train $EPOCH_EVAL_TRAIN \
    --eval_eval_freq $EVAL_EVAL_FREQ \
    --dataset $DATASET \
    --image_encoder $IMAGE_ENCODER \
    --text_encoder $TEXT_ENCODER \
    --name $NAME \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_test $BATCH_SIZE_TEST
