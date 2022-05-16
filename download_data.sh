#!/usr/bin/bash
# Usage: change pwd to <project root dir> and execute `bash get_data.sh landscape`
# this script is partly borrowed from https://github.com/Jittor/gan-jittor/blob/master/data/download_pix2pix_dataset.sh

FILE=$1
echo $FILE

TRAIN_URL=https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1
TRAIN_ZIP=train.zip
wget -N $TRAIN_URL -O $TRAIN_ZIP

TEST_ZIP=test.zip
TEST_URL=https://cloud.tsinghua.edu.cn/f/70195945f21d4d6ebd94/?dl=1
wget -N $TEST_URL -O $TEST_ZIP

TARGET_DIR=datasets/$FILE
rm -rf $TARGET_DIR
mkdir $TARGET_DIR

unzip -q $TRAIN_ZIP -d ./$TARGET_DIR/
rm $TRAIN_ZIP

unzip -q $TEST_ZIP -d ./$TARGET_DIR/
mv -f $TARGET_DIR/val_A_labels_cleaned $TARGET_DIR/test
rm $TEST_ZIP

rm -rf $TARGET_DIR/val
mkdir -p $TARGET_DIR/val/imgs $TARGET_DIR/val/labels
