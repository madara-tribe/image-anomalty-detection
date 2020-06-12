from VAE import VAE
from train import trainGenerator
import seaborn as sns
import pandas as pd
from train import load_dataset

def plot_loss(normal_loss, anomaly_loss):
    test_anodf = pd.DataFrame(np.array(normal_loss), columns=['anomaly'])
    test_ndf = pd.DataFrame(np.array(anomaly_loss), columns=['normal'])
    print(test_anodf.shape, test_ndf.shape)

    # plot 
    sns.distplot(test_anodf["anomaly"], kde=True, rug=True, color = 'red', label = 'test_anomaly')
    sns.distplot(test_ndf["normal"], kde=True, rug=True, color = 'green', label = 'test_normal')
    plt.legend()
    plt.show()


def anomaly_detector(model, x_normal, x_anomaly):
    normal_loss =[]
    anomaly_loss=[]
    #正常のスコア
    for img in x_normal:
      mu, sigma = model.predict(img.reshape(1, 256, 256, 1), batch_size=1, verbose=0)
      # not img but x_normal
      nloss =(0.5 * (x_normal - mu)**2) / sigma
      normal_loss.append(sum(nloss.flatten()))
    
    #異常のスコア
    for ano in x_anomaly:
      mu, sigma = model.predict(ano.reshape(1, 256, 256, 1), batch_size=1, verbose=0)
      # not ano but x_anomaly
      ano_loss =(0.5 * (x_anomaly - mu)**2) / sigma
      anomaly_loss.append(sum(ano_loss.flatten()))
 
    plot_loss(normal_loss, anomaly_loss)

def main():
    test_anomaly = load_dataset('anomaly')
    test_normal = load_dataset('normal')

    model = VAE()
    model, loss = model.vae_net()
    model.load_weights("weight/vae_model.h5")
    anomaly_detector(model, test_normal, test_anomaly)



if __name__ == '__main__':
    main()