from sklearn.covariance import EmpiricalCovariance, MinCovDet


"""each MCD(マハラノビス距離) ano score")"""
def MCD_Score(train_a, test_a, test_b):
    mcd = MinCovDet()
    mcd.fit(train_a)
    mcd_anoscore = mcd.mahalanobis(test_a)
    mcd_normalscore = mcd.mahalanobis(test_b)
    print("mcd ano score {} mcd normal score {}".format(mcd_anoscore, mcd_normalscore))
