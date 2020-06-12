import keras 
from keras import Input
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2


# metric learning model based InceptionResNetV2
def metric_model(load_weight=False, weight_path=None):
    alpha=5

    input_tensor = Input(shape=(256, 256, 3))
    model = InceptionResNetV2(include_top=True, weights=None, input_tensor=input_tensor, pooling="avg", classes=2)
    print(model.layers[-3].output)
    c = GlobalAveragePooling2D()(model.layers[-3].output)
    c = Lambda(lambda xx: alpha*(xx)/K.sqrt(K.sum(xx**2)))(c) #metric learning
    c = Dense(2, activation='softmax')(c)
    metric_model = Model(inputs=model.input, outputs=c)
    if load_weight:
        metric_model.load_weights(weight_path)
    metric_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True), metrics=['accuracy'])  
    metric_model.summary()
    return metric_model
