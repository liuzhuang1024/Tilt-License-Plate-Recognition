#!/usr/bin/env bash
GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_nips /home/lz/CCPD2019/new_train/all_train \
	--train_cvpr /home/lz/CCPD2019/new_train/all_train \
	--valroot /home/lz/CCPD2019/new_train/all_test \
	--workers 2 \
	--batchSize 32 \
	--niter 10 \
	--lr 1 \
	--cuda \
	--experiment output/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder