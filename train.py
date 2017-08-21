from eval_tools import evaluation_all,evaluation_with_distance
from geo_tweets_data_fs import geo_data_clean,geo_dataset_pre
from model.rnn_model_keras import geo_lprnn_model
import config
import numpy as np
GRID_COUNT = config.GRID_COUNT

def geo_rnn_train(train_X, train_Y, vali_X, vali_Y,vali_evl, model,center_location_list, epoch=60):

    for i in range(epoch):
        print i
        model.fit(train_X, train_Y,epochs=1,batch_size=50, validation_data=(vali_X,vali_Y))
        if (i % 5) == 0:
            all_output_array = model.predict(vali_X)
            evaluation_with_distance(all_output_array,vali_evl,center_location_list)
            model.save('./model/User_RNN_Seg_Epoch_0.001_100_rmsprop_'+str(i)+'.h5')


if __name__ == '__main__':

    user_feature_sequence, place_index, seg_max_record, center_location_list = geo_data_clean()
    print len(user_feature_sequence.keys())
    train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl,user_dim\
        = geo_dataset_pre(user_feature_sequence,seg_max_record)
    # print ('x shape',train_X.shape)
    model = geo_lprnn_model(user_dim,seg_max_record)
    geo_rnn_train(train_X, train_Y,vali_X, vali_Y,vali_evl, model, center_location_list)
