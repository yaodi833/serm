from geo_data_decoder import *
from eval_tools import *
import config
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]  = config.GPU

PRETRAINED_FS = config.PRETRAINED_FS
PRETRAINED_LA = config.PRETRAINED_LA

def train(dataset = 'FS'):

    if dataset == 'FS':
        user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec = geo_data_clean_fs()
        print len(user_feature_sequence.keys())
        train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index \
            = geo_dataset_train_test_text(user_feature_sequence, useful_vec, seg_max_record)
        print ("Feature generation completed")
        nearest_location_last(vali_X, vali_evl, center_location_list)
        model = geo_lprnn_trainable_text_model(user_dim, seg_max_record, word_vec)
        model.load_weights(PRETRAINED_FS)
        all_output_array = model.predict(vali_X)
        evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
        print ("Train_x[0] shape:", train_X[1].shape)
        print ("Train_x[0] shape:", train_X[2].shape)
        print ("Train_Y shape:", train_Y.shape)
        geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y, vali_evl, model, center_location_list, word_index,
                                 dataset='FS_trainable_')

    elif dataset=='LA':
        user_feature_sequence, place_index, seg_max_record, center_location_list, useful_vec= geo_data_clean_la()
        print len(user_feature_sequence.keys())

        train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec, word_index\
            = geo_dataset_train_test_text(user_feature_sequence,useful_vec, seg_max_record)
        print ("Feature generation completed")
        frequent_location_last(train_X, vali_X, vali_evl, center_location_list)
        nearest_location_last(vali_X, vali_evl, center_location_list)
        model =geo_lprnn_trainable_text_model(user_dim,seg_max_record,word_vec)
        model.load_weights(PRETRAINED_LA)
        all_output_array = model.predict(vali_X)
        evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
        print ("Train_x[0] shape:", train_X[1].shape)
        print ("Train_x[0] shape:", train_X[2].shape)
        print ("Train_Y shape:", train_Y.shape)
        geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y, vali_evl, model, center_location_list,word_index,
                                 dataset='LA')



if __name__ == '__main__':
    train(dataset='FS')