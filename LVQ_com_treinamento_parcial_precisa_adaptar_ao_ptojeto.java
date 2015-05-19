//nao estou usando o JAMA

import java.util.Random;
import java.util.List;

class LVQ{
	int [][] vetores_de_entrada;	//dados de entrada
	int numero_de_instancias;		//vetores_de_entrada[i]
	int numero_neuronios_saida;		//vetores prototipos
	int dimensao;					//da entrada
	double taxa_de_aprendizado;		//definir
	double[][]pesos;				//aleatorio
	int neuronio_ganhador;			//indice do vetor da classe ganahdora
	int[] classe_alvo;	//			//vetor que guarda as classes das instancias
	int numero_de_classes;			//alvo
	double[] distancia_euclidiana;	//conterah as distancias de uma entrada em relacao aos neuronios
	int iteracoes;					//parametro de treinamento
	List<Integer> erro;
	
	public LVQ(int [][] vetores_de_entrada, int numero_de_instancias, int numero_neuronios_saida, double taxa_de_aprendizado, int[]classe_alvo, int numero_de_classes, int dimensao){	//tem outros atributos
		this.vetores_de_entrada = vetores_de_entrada;
		this.numero_de_instancias = numero_de_instancias;
		this.numero_neuronios_saida = numero_neuronios_saida;
		this.taxa_de_aprendizado = taxa_de_aprendizado;
		//this.pesos = pesos;
		this.classe_alvo = classe_alvo;
		this.numero_de_classes = numero_de_classes;
		this.dimensao = dimensao;
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
	
	void imprime_matriz_de_pesos(){
		for(int x=0; x < numero_neuronios_saida; x++){    
			for(int y=0; y < dimensao; y++){
                System.out.print(pesos[x][y]+"\t");
            }
            System.out.println();
        }
	}
	
	/* A constante pode ser trocada, porem como visto em aula, 0.9 eh um bom numero */
	
	public double diminui_taxa_de_aprendizado(double taxa_de_aprendizado_atual){
		double taxa_atualizada;
		taxa_atualizada = taxa_de_aprendizado_atual*0.9;
		return taxa_atualizada;
	}
	
	
	/* metodo para guardar a classe de cada instancia na matriz de entrada */
	
	//public void define_classe(){
		//classe_alvo
		// Definir o numero de neuronios de saida e suas respectivas classes

	//}
	
	
	/* o treino pode ter como condicao de parada o numero de iteracoes (epocas), valor minimo da taxa de aprendizado,
	erro de quantizacao maximo, etc. O algoritmo a seguir eh baseado nos slides que o Antonio achou da professora Sara */
	
	void treina_lvq(int numero_de_iteracoes){	
			//List<Integer> erro_quadratico_medio = new ArrayList<Integer>();
			distancia_euclidiana = new double[numero_neuronios_saida];	//matriz que guardarah as distancias de uma instancia em relacao a cada vetor prototipo
			/* Passo 1 - for determinado pelo numero de iteracoes, que eh um parametro 
			Enquanto a condicao de parada nao eh alcancada, execute os passos 2 - 6	*/
			for(int epoca = 0; epoca < numero_de_iteracoes; epoca++){
				System.out.println("Epoca: " + epoca);
				/* Passo 2 - Para cada vetor de entrada do treinamento, executa os passos 3 e 4 */
				for(int k = 0; k < numero_de_instancias; k++){	
					/* Passo 3 - Encontrar a unidade(neuronio) de saida tal que a distancia seja minima */
					neuronio_ganhador = 0;
					for(int i = 0; i< numero_neuronios_saida; i++){
						distancia_euclidiana[i] = distancia_euclidiana(pesos[i] ,vetores_de_entrada[k]);
						if(i!= 0){
							if(distancia_euclidiana[i]<distancia_euclidiana[neuronio_ganhador]){
								neuronio_ganhador = i;
							}
						}
						System.out.println(distancia_euclidiana[i] + "	neuronio ganhador:" + neuronio_ganhador);
					}
					/* Passo 4 - Alterando os pesos analisando a classe definida. Sao aplicados as regras de aprendizado. A soma ocorre quando 
					a classificacao sugerida pela LVQ eh correta (o vetor resultante da operação está situado entre o vetor protótipo e o vetor
					de dados, ou seja, houve a movimentação do vetor protótipo na direção do dado). Já a subtracao ocorre quando a classificacao
					sugerida eh incorreta (tendo efeito contrario)*/
					if(classe_alvo[k] == neuronio_ganhador){	// Preciso colocar os rotulos dos pesos
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
			/* Eliminar neuronios nao utilizados */
	}
	
	/* Testando o XOR */
	

//ublic LVQ(int [][] vetores_de_entrada, int numero_de_instancias, int numero_neuronios_saida, double taxa_de_aprendizado, int[]classe_alvo, int numero_de_classes, int dimensao, )

	public static void main(String[]args){
		int[][] entrada = new int[4][2];
		entrada[0][0] = 1;
		entrada[0][1] = 1;
		
		entrada[1][0] = 0;
		entrada[1][1] = 0;
		
		entrada[2][0] = 1;
		entrada[2][1] = 0;
		
		entrada[3][0] = 0;
		entrada[3][1] = 1;
		
		int[] classes = new int[4];
		classes[0] = 0;
		classes[1] = 0;
		classes[2] = 1;
		classes[3] = 1;
		
		LVQ lvq_teste = new LVQ(entrada, 4, 100, 0.1, classes, 2, 2 );
		lvq_teste.inicializa_pesos();
		System.out.println("Matriz de pesos:");
		System.out.println();
		lvq_teste.imprime_matriz_de_pesos();
		System.out.println();
		System.out.println("Treinando a LVQ!!!!");
		System.out.println();
		lvq_teste.treina_lvq(1);
		
	}
}
