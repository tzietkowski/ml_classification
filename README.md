# Summary model:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       

 max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     

 max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0
 2D)

 dropout (Dropout)           (None, 6, 6, 64)          0

 flatten (Flatten)           (None, 2304)              0

 dense (Dense)               (None, 128)               295040    

 dense_1 (Dense)             (None, 4)                 516

=================================================================
Total params: 314,948
Trainable params: 314,948
Non-trainable params: 0
```
# Result fit model:
Accuracy:
<img width="476" alt="image" src="https://user-images.githubusercontent.com/73225607/230617692-53080ec1-aad8-4f76-9f31-f07650cb82b2.png">

Loss:
<img width="475" alt="image" src="https://user-images.githubusercontent.com/73225607/230617743-7cc5ce17-cb06-4c5d-8a4e-2d50fd07267b.png">

# Results for test data:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       133
           1       1.00      1.00      1.00       110
           2       1.00      1.00      1.00       141
           3       1.00      1.00      1.00        96

    accuracy                           1.00       480
   macro avg       1.00      1.00      1.00       480
weighted avg       1.00      1.00      1.00       480
```
# Prediction result for 10 random images:
<img width="688" alt="image" src="https://user-images.githubusercontent.com/73225607/230618199-dd69eb89-f191-4dcf-8014-f73c7f9ee5aa.png">
