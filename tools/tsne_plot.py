"""
Plot anomaly and generated image by TSNE
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_plot(normal_img, anomaly_query):
    
    #normal_feature_map = [cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY) for imgs in feature_map_of_random]
    normal_feature_map = np.array(normal_img)
    # anomaly_query
    #anomaly_feature_map = [cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY) for imgs in anomaly_query]
    anomaly_feature_map = np.array(anomaly_query)

    # t-SNE for visulization
    output = np.concatenate((normal_feature_map, anomaly_feature_map))
    output = output.reshape(output.shape[0], -1)
    
    nbatch = len(normal_img)
    X_embedded = TSNE(n_components=2).fit_transform(output)
    print(X_embedded.shape)
    plt.figure(5)
    plt.title("t-SNE embedding on the feature representation")
    plt.scatter(X_embedded[:nbatch,0], X_embedded[:nbatch,1], label='normal', c='blue')
    plt.scatter(X_embedded[nbatch:,0], X_embedded[nbatch:,1], label='anomaly', c='red')
    plt.legend()
    plt.show()