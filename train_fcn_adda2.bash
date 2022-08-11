
######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=100000
crop=768
snapshot=5000
batch=1


resdir="results/cyclegta5_to_cityscapes/adda_sgd/discrim_score_nolsgan_${discrim}"
base_model="base_models/drn26-cyclegta5-iter115000.pth"
outdir="${resdir}/drn26/lr1e-5_crop${crop}_ld1_lg0.1_momentum0.99"

# Run python script #
CUDA_VISIBLE_DEVICES=8 python scripts/train_fcn_adda.py \
    ${outdir} \
    --dataset 'cyclegta5' --dataset 'cityscapes' --datadir '/output/' \
    --lr 1e-5 --momentum 0.99 --gpu 0 \
    --lambda_d 1 --lambda_g 0.1 \
    --weights_init ${base_model} --model 'drn26' \
    --"'discrim_score'" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot
