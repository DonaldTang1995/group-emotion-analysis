for i in 1
do
python extract.py --net_number $i --test_set train --loss softmax --model_path "weights/"$i"nets_softmax_model.npy"     --data_path ../train/
python LSTM_train.py --net_number  $i --train_path "features/train_"$i"nets_feature.npy" --val_path "features/val_"$i"nets_feature.npy" --iter 4
done
