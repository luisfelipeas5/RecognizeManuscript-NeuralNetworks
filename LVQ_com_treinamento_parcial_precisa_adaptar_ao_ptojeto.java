//nao estou usando o JAMA

import java.util.Random;

class LVQ{
	int [][] vetores_de_entrada;	//dados de entrada
	int numero_de_instancias;
	int numero_neuronios_saida;
	int dimensao;					//da entrada
	double taxa_de_aprendizado;
	double[][]pesos;
	int neuronio_ganhador;			//vetor que possui a classe ganahdora
	int[] classe_alvo;	//		
	int numero_de_classes;
	double[] distancia_euclidiana;	//conterah as distancias de uma entrada em relacao aos neuronios
	int iteracoes;
	
	
	public LVQ(int numero_de_classes, int dimensao, double taxa_de_aprendizado){	//tem outros atributos
		this.numero_de_classes = numero_de_classes;
		this.dimensao = dimensao;
		this.taxa_de_aprendizado = taxa_de_aprendizado;
	}
	
	
	/* Distancia Euclidiana sera utilizada no EP */
		
	public double distancia_euclidiana(double[] vetor1, int[] vetor2){
		double distancia =0;
		for(int j=0;j<dimensao;j++){
			distancia = distancia + Math.pow((vetor1[j] - vetor2[j]), 2);
		}
		return Math.sqrt(distancia);
	}
	
	/* A matriz de pesos tem o numero de linhas igual ao numero de neuronios de saida
	e numero de colunas igual ao numero de atributo, ou seja, a dimensao das entrasas */
	
	public void inicializa_pesos(){	//aleatorio
		pesos = new double[numero_neuronios_saida][dimensao];
		Random rd = new Random();
		for(int i = 0; i<numero_neuronios_saida; i++){
			for(int j = 0; j<dimensao; j++){
				pesos[i][j] = rd.nextDouble();
			} 	
		}	
	}
	
	/* A constante pode ser trocada, porem como visto em aula, 0.9 eh um bom numero */
	
	public double diminui_taxa_de_aprendizado(double taxa_de_aprendizado_atual){
		double taxa_atualizada;
		taxa_atualizada = taxa_de_aprendizado_atual*0.9;
		return taxa_atualizada;
	}
	
	
	/* metodo para guardar a classe de cada instancia na matriz de entrada */
	
	public void define_classe(){
		//classe_alvo
		// Definir o numero de neuronios de saida e suas respectivas classes
		
	}
	
	
	/* o treino pode ter como condicao de parada o numero de iteracoes (epocas)ou valor minimo da taxa de aprendizado
	O algoritmo a seguir eh baseado nos slides que o Antonio achou da professora Sara */
	
	void treina_lvq(int numero_de_iteracoes){	
			distancia_euclidiana = new double[numero_de_classes];	//matriz que guardarah as distancias de uma instancia em relacao a cada vetor prototipo
			/* Passo 1 - for determinado pelo numero de iteracoes, que eh um parametro 
			Enquanto a condicao de parada nao eh alcancada, execute os passos 2 - 6	*/
			for(int epoca = 0; epoca < numero_de_iteracoes; epoca++){
				/* Passo 2 - Para cada vetor de entrada do treinamento, executa os passos 3 e 4 */
				for(int k = 0; k < numero_de_instancias; k++){	
					/* Passo 3 - Encontrar a unidade(neuronio) de saida tal que a distancia seja minima */
					neuronio_ganhador = 0;
					for(int i = 0; i< numero_neuronios_saida; i++){
						distancia_euclidiana[i] = distancia_euclidiana(pesos[i] ,vetores_de_entrada[i]);
						if(i!= 0){
							if(distancia_euclidiana[i]<distancia_euclidiana[neuronio_ganhador]){
								neuronio_ganhador = i;
							}
						}
						//System.out.println(distancia_euclidiana[i] + "neuronio ganhador:" + neuronio_ganhador);
					}
					/* Passo 4 - Alterando os pesos analisando a classe definida. Sao aplicados as regras de aprendizado. A soma ocorre quando 
					a classificacao sugerida pela LVQ eh correta (o vetor resultante da operação está situado entre o vetor protótipo e o vetor
					de dados, ou seja, houve a movimentação do vetor protótipo na direção do dado). Já a subtracao ocorre quando a classificacao
					sugerida eh incorreta (tendo efeito contrario)*/
					if(classe_alvo[k] == neuronio_ganhador){
						for(int a = 0; a < dimensao; a++){
							pesos[neuronio_ganhador][a] = pesos[neuronio_ganhador][a] + (taxa_de_aprendizado * (vetores_de_entrada[k][a] - pesos[neuronio_ganhador][a])); 
						}
					}else{
						for(int a = 0; a < dimensao; a++){
							pesos[neuronio_ganhador][a] = pesos[neuronio_ganhador][a] - (taxa_de_aprendizado * (vetores_de_entrada[k][a] - pesos[neuronio_ganhador][a])); 
						}
					}	
				}
				/* Reduzir a taxa de aprendizado, pode ser por meio de uma constante (sendo linear) ou por meio de uma funcao */
				taxa_de_aprendizado = diminui_taxa_de_aprendizado(taxa_de_aprendizado);	//System.out.println(taxa_de_aprendizado);
			}	/* Passo 6 - Continua testando a condicao de parada */	
	}
	
	/* Testando o XOR */
	
	public static void main(String[]args){
		System.out.println("Treinando a LVQ!!!!");
		
	}
}
