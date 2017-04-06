import tensorflow as tf
import numpy as np
import argparse
from LSTM import LSTM
def load_data(path,feature_path,flag=False):
    data_=np.load(path)
    features=None
    if feature_path!=None:
        features=np.load(feature_path).item()
    data=[]
    target=[]
    features_=[]
    total=0
    for instance in data_:
        if features!=None and instance['name'] not in features:
            continue
        if instance['label']==-1:
            if not flag:
                continue
            else:
                data.append(np.array(instance['feature']))
                target.append([])
                if features!=None:
                    features_.append(np.reshape(features[instance['name']],(1,1176)))
        else:
            data.append(np.array(instance['feature']))
            target.append(instance['label'])
            if features!=None:
                features_.append(np.reshape(features[instance['name']],(1,1176)))
                
    print total
    return data,target,features_

def parse_args():
    parser=argparse.ArgumentParser(description='group emotion analysis')
    parser.add_argument('--net_number',dest='number_of_nets',default=1,type=int)
    parser.add_argument('--train_path',dest='train_path')
    parser.add_argument('--val_path',dest='val_path')
    parser.add_argument('--iter',dest='max_step',type=int)
    parser.add_argument('--weights',dest='weights')
    parser.add_argument('--geo_feature_path',dest='geo_feature_path')
    return parser.parse_args()    

def main():
    
    args=parse_args()
    model=LSTM(args.number_of_nets)
    x=[]
    if args.geo_feature_path!=None:
        x.append(tf.placeholder(tf.float32,(1,1176)))
    model.inference(x)

    if args.weights!=None:
        model.assign_weights(args.weights)

    train_data,train_target,train_features=load_data(args.train_path,args.geo_feature_path)
    train_data1,_,train_features1=load_data(args.train_path,args.geo_feature_path,True)
    val_data,val_target,val_features=load_data(args.val_path,args.geo_feature_path)
    val_data1,_,val_features1=load_data(args.val_path,args.geo_feature_path,True)
    for i in xrange(args.max_step):
        model.train(train_data,train_target,x,train_features)
    model.save('weights/%dnets_group_label_lstm.npy'%(args.number_of_nets))
    model.test(val_data,val_target,x,val_features)
    features=model.extract(train_data1,x,train_features1)    
    features_={}
    for f,feature in zip(np.load(args.train_path),features):
        features_[f['name']]=feature
    print len(features_)
    np.save('features/train_{}nets_LSTM_features.npy'.format(args.number_of_nets),features_)
    features=model.extract(val_data1,x,val_features1)
    features_={}
    for f,feature in zip(np.load(args.val_path),features):
        features_[f['name']]=feature
    print len(features_)
    np.save('features/val_{}nets_LSTM_features.npy'.format(args.number_of_nets),features_)
if __name__=='__main__':
    main() 
