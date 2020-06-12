from styleGAN.styleGAN import styleGAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

GAN = styleGAN()
D = GAN.discriminator()
D.load_weights("styleGAN/tb/discriminator.h5")

""" caliculate anomaly score """

def LOF_Score(trains, test_anomaly, test_normal):
    train_a = D.predict(trains) # train images
    test_a = D.predict(test_anomaly) # test anomaly
    test_b = D.predict(test_normal) # test normal

    train_a = train_a.reshape((len(trains),-1))
    test_a = test_a.reshape((len(test_anomaly),-1))
    test_b = test_b.reshape((len(test_normal),-1))
    print(train_a.shape, test_a.shape, test_b.shape)

    # MinMaxScaler
    ms = MinMaxScaler()
    train_a = ms.fit_transform(train_a)
    test_a = ms.transform(test_a)
    test_b = ms.transform(test_b)
    clf = LocalOutlierFactor(n_neighbors=5)
    clf.fit(train_a)


    # caliculate anomaly score
    Z1 = -clf._decision_function(test_a)
    Z2 = -clf._decision_function(test_b)
    print('ano score {}, normal score {}'.format(sum(Z1), sum(Z2)))
    return train_a, test_a, test_b


def lof_each_ano_score(train_a, test_a, test_b):
    lof = LocalOutlierFactor(n_neighbors=5,
                           novelty=True,
                           contamination=0.1)

    lof.fit(train_a)
    # each LOF prediction label (-1 is anomaly and 1 is normal)
    test_a_pred = lof.predict(test_a) # テストデータに対する予測
    test_b_pred = lof.predict(test_b)
    print(test_a_pred, test_b_pred)