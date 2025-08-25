from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''chama as bibliiotecas: numpy para criar a array, seaborn e matplot para criar o gráfico e o metodo confusion_matrix do sklearn matrix para criar a matriz de confusão'''

y_true = np.random.randint(0, 10, 100)
y_pred = y_true.copy()

'''cria os arrays true, para que possa pegar os dados e o pred, que é uma cópia, a fim de alterar os dados e comparar com o true'''

indices = np.random.choice(len(y_pred), size=50, replace=False)
y_pred[indices] = (y_pred[indices] + np.random.randint(1, 10, size=50)) % 10
'''É selecinoado 50 números de maneira aleatória para simular os erros, usa o replace = False para não pegar repetidos, depois altera os indeices de maneira aleatória e é dividido por 10 para garrantir que fique entra 0 e 9'''



cm = confusion_matrix(y_true, y_pred, labels=np.arange(10), normalize='true')
'''metodo confusion_matrix recebendo as arrays, definindo seu tamanho e normalizando para os valores virarem porcentagem, a fim de não parecer que um é melhor que o outro apenas por causa de sua maior quantidade'''


'''mapa de calor com o annot = true para colocar os números nos quadrados'''

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.title("Matriz de Confusão Normalizada 10x10")
plt.show()

class Metricas():
  def __init__(self, vp, vn, fp, fn):
    '''classe que recebe os dados de vp, vn, fp, fn no construtor'''
    self.vp = vp
    self.vn = vn
    self.fn = fn
    self.fp = fp
    self.n = vp + vn + fp + fn

  def sensibilidade(self):
    '''função que calcula a sensibilidade'''
    s = (self.vp)/(self.vp + self.fn)
    return s

  def especificidade(self):
    '''função que calcula a especificidade'''
    e = (self.vn)/ (self.fp + self.vn)
    return e

  def acuracia(self):
    '''função que calcula a acurácia'''
    a = (self.vp + self.vn)/(self.n)
    return a
  
  def precisao(self):
    '''função que calcula a precisão'''
    p = (self.vp) / (self.vp + self.fp)
    return p
 
  def fscore(self):
    '''função que calcula a f-score'''
    f_score = 2*(self.precisao()*self.sensibilidade())/(self.precisao()+ self.sensibilidade())
    return f_score

def main():
   '''função main que utiliza de um loop for para percorrer as colunas da matriz e identificar cada dado'''

   for i in range(10):
    vp = cm[i, i]
    fn = np.sum(cm[i, :]) - vp
    fp = np.sum(cm[:, i]) - vp
    vn = np.sum(cm) - (vp + fn + fp)
 
   resultado = Metricas(vp, vn, fp, fn)

   print(f'Sensibilidade: {resultado.sensibilidade()}')
   print(f'Especificidade: {resultado.especificidade()}')  
   print(f'Acurácia: {resultado.acuracia()}')
   print(f'Precisão: {resultado.precisao()}')  
   print(f'F-score: {resultado.fscore()}')  


if __name__ == '__main__':
  main()
