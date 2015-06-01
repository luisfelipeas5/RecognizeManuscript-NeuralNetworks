# EP1-IA
Repositório destinado ao EP 1 de Inteligência Artificial.

Grupo:
-André Mello
-Antonio Mateus
-Luís Felipe de Almeida da Silva
-Marcelo Kazuya Kajiwara

Parâmetros para serem passados através da linha de comando
#Entrada padrao:
	1. Nome do arquivo do conjunto de dados de treino.
	2. Nome do arquivo do conjunto de dados de validacao.
	3. Nome do arquivo do conjunto de dados de teste.
	4. Taxa de aprendizado inicial.
	5. Numero de neuronios de camada escondida (MLP).
	6. Numero de neuronios para a camada classe (LVQ).
	7. Inicializacao de pesos:
	  -> Inserir 'z' para matriz de pesos 0 ou 'a' para matriz de pesos aleatorios.
#Entrada padrao modificada:
	1. Nome do arquivo com todos os dados
	2. Taxa de aprendizado inicial
	3. Numero de neuronios de camada escondida (MLP)
  4. Numero de neuronios para a camada classe (LVQ)
  5. Inicializacao de pesos:
      -> Inserir 'z' para matriz de pesos 0 ou 'a' para matriz de pesos aleatorios.
#Entrada extendida:
	1. Nome do arquivo com todos os dados.
      -> Inserir: 'nenhum' ou 'max' ou 'minmax' ou 'zscore' ou 'sigmoidal'.
	2. Tipo de reducao de dados: remover colunas preenchidas com zeros ou colunas com baixo desvio padrao:
      -> Inserir: 'nenhum' ou 'zeros' ou 'desvio'.
	3. Valor de corte:
    3.1. Em caso de eleminacao de colunas por atributos com zeros:
      -> Inserir valor entre 0 e 100, correspondente a porcentagem de zeros que um atributo deve ter para ser eliminado.
    3.2. Em caso de eliminacao de colunas por atributos com baixo desvio padrao:
      -> Inserir valor minimo de desvio padrao desejado para as colunas. Colunas que com desvio padrao menor ou igual ao inserido serao eliminadas.
    3.3. Em caso do paramtro 3 = 'nenhum':
      -> Inserir: 0.
  4. Tipo de normalizacao:
	5. Taxa de aprendizado inicial.
	6. Numero de neuronios de camada escondida (MLP)
	7. Tipo de treinamento da MLP:
      -> Inserir: 'batelada' para treinamento em batelada ou 'padrao' para treinamento padrao a padrao.
	8. Numero de neuronios para a camada classe (LVQ)
	9. Inicializacao de pesos:
      -> Inserir 0 para se utilizar uma matriz de pesos 0 ou inserir qualquer outro numero para criar uma matriz de pesos aleatorios com valores variando -x e x, sendo x o valor passado via este parametro.
	10. Limiar de erro. Ao atingir esse valor a rede ira encerrar seu treinamento.
	11 . Numero maximo de epocas. Ao atingir esse valor a rede ira encerrar seu treinamento.
