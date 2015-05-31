import Jama.Matrix;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LVQ extends Rede{
	
	/* Atributos de entrada */
	double[][] vetores_de_entrada;	// Os dados de entrada serao armazenadas nessa matriz
	double[] classe_alvo;			// Vetor que guarda as classes das instancias (de entrada), ou a saida desejada
	int numero_de_instancias;		// Quantidade de linhas da matriz de entrada (numero de entradas)
	int dimensao;					// Quantidade de colunas da matriz de entradas (atributos)
	int numero_de_classes;			// Quantidade de classes possiveis para classificacao (no caso do EP, 10)
	
	/* Atributos relacionados aos pesos */
	double[][]pesos;				// Pesos que sao passados pela classe Treinamento (cada linha representa um neurônio)
	double[] rotulo_pesos;			// Classes dos neuronios (onde cada linha de pesos[][] tem sua classe na mesma linha em rotulo_pesos[])
	
	/* Parametros e atributos para o funcionamento da rede */
	int numero_neuronios_saida;		// Quantidade de neuronios (vetores prototipos) para cada classe
	double taxa_de_aprendizado;		// Taxa que determina a regulacao de convergencia
	int neuronio_ganhador;			// Variavel que guarda o indice do neuronio ganhador (dentre todos os vetores)
	double[] distancia_euclidiana;	// conterah as distancias de uma entrada em relacao aos neuronios
	
	boolean necessidade_atualizar_pesos = true;	// Controle da classe Treinamento.java 
	
	/* Construtor da rede LVQ 
	 * Estende a classe abstrata rede, que tambem possue construtor */
	public LVQ(int numero_neuronios_classe, double taxa_de_aprendizado, double[] classes){	
		super(numero_neuronios_classe);
		this.numero_neuronios_saida = numero_neuronios_classe;
		this.taxa_de_aprendizado = taxa_de_aprendizado;
		this.numero_de_classes = classes.length;
		
		/* Os rotulos sao inicializados de acordo com a quantidade de neuronios para cada classe */
		Arrays.sort(classes);
		this.rotulo_pesos=new double[numero_neuronios_classe*classes.length];
		int j=0;
		for(int i=0; i < classes.length;i++) {
			for (int k=0; k < numero_neuronios_classe; k++) {
				this.rotulo_pesos[j]=(double)classes[i];
				j++;
			}
		}
	}
	
	/* As matrizes de entrada e saida desejada sao copiadas para os atributos desta classe
	 * As copias sao realizadas por conveniencia (e familiarizacao com java)  */
	void set_problema (Matrix entrada, Matrix saida_desejada){
		this.vetores_de_entrada = entrada.getArrayCopy();
		this.numero_de_instancias = entrada.getRowDimension();
		this.dimensao = entrada.getColumnDimension();
		this.classe_alvo = saida_desejada.getRowPackedCopy();
	}
	
	/* Os vetores prototipos (ou neuronios) sao dados para a rede */ 
	void set_pesos (Matrix pesos_a, Matrix pesos_b){
		this.pesos = pesos_a.getArrayCopy();
	} 
	
	void set_modo_treinamento (int modo_treinamento){
		// Utiliza-se apenas na rede MLP
	} 
	
	/* Metodo que calcula a Distancia Euclidiana, que sera utilizada no EP */
	public double distancia_euclidiana(double[] vetor1, double[] vetor2){
		double distancia =0;
		for(int j=0;j<dimensao;j++){
			distancia = distancia + Math.pow((vetor1[j] - vetor2[j]), 2);
		}
		return Math.sqrt(distancia);
	}
	
	/* Imprime a matriz de pesos */
	void imprime_matriz_de_pesos(){
		for(int x=0; x < numero_neuronios_saida*numero_de_classes; x++){    
			for(int y=0; y < dimensao; y++){
                System.out.print(pesos[x][y]+"\t");
            }
            System.out.println();
        }
	} 
	
	/* Imprime a classe de cada vetor prototipo (neuronio) */
	void imprime_rotulos_dos_pesos(){
		for (int y = 0; y < rotulo_pesos.length; y++){
			System.out.println(rotulo_pesos[y]);
		}	
	}
	
	/* Metodo responsavel pela reducao da taxa de aprendizado
	 * A diminuicao ocorre no final de cada epoca do treinamento da rede */
	public double diminui_taxa_de_aprendizado(double taxa_de_aprendizado_atual){
		double taxa_atualizada;
		taxa_atualizada = taxa_de_aprendizado_atual*0.9999;
		return taxa_atualizada;
	}

	/* Retorna o erro de um unica epoca, ou seja, a quantidade de vezes que o neuronio ganhador 
	 * nao era da mesma classe que a instancia dividido pelo numero de instancias.
	 * Nesse metodo, eh feito uma epoca da rede LVQ, seu treinamento eh controlado pela classe
	 * Treinamento.java */
	double get_erro(){	
		double contador_de_erros = 0;	// Sera retornado ao final de uma epoca
		
		/* Matriz que guardarah as distancias de uma instancia em relacao a cada vetor prototipo */
		distancia_euclidiana = new double[numero_neuronios_saida*numero_de_classes];
		
		/* Passo 1 - Para cada vetor de entrada, executa os proximos dois passos (3 e 4) */
		for(int k = 0; k < numero_de_instancias; k++){	

			/* Passo 2 - Encontrar o vetor (neuronio) de saida tal que a distancia seja a menor */
			neuronio_ganhador = 0;
			for(int i = 0; i< numero_neuronios_saida*numero_de_classes; i++){
				distancia_euclidiana[i] = distancia_euclidiana(pesos[i] ,vetores_de_entrada[k]);
				if(i!= 0){
					if(distancia_euclidiana[i]<distancia_euclidiana[neuronio_ganhador]){
						neuronio_ganhador = i;
					}
				}
			}
		
			/* Se os pesos devem ser atualizados (isto eh, se o conjunto de entradas for o conjunto de treinamento), atualiza.
			 * Nao ha atualizacao de pesos quando temos o conjunto de validacao como conjunto de entradas).
			 */
			if(necessidade_atualizar_pesos) {
				
				/* Passo 3 - Alterando os pesos analisando a classe definida. Sao aplicados as regras de aprendizado. 
				 * A soma ocorre quando a classificacao sugerida pela LVQ eh correta (o vetor resultante da operacaoo
				 *  esta situado entre o vetor prototipo e o vetor de dados, ou seja, houve a movimentacao do vetor 
				 *  prototipo na direcaoo do dado). Ja a subtracao ocorre quando a classificacao sugerida eh incorreta
				 *  (tendo efeito contrario) e o contador de erros eh incrementado */
				//imprime_rotulos_dos_pesos();
				if(classe_alvo[k] == rotulo_pesos[neuronio_ganhador]){	// Compara a classe da entrada com a do neuronio
					//System.out.println("Acertou!!");
					for(int a = 0; a < dimensao; a++){
						pesos[neuronio_ganhador][a] = pesos[neuronio_ganhador][a] + (taxa_de_aprendizado * (vetores_de_entrada[k][a] - pesos[neuronio_ganhador][a])); 
					}
				}else{
					//System.out.println("Errou!");
					contador_de_erros++;
					for(int a = 0; a < dimensao; a++){
						pesos[neuronio_ganhador][a] = pesos[neuronio_ganhador][a] - (taxa_de_aprendizado * (vetores_de_entrada[k][a] - pesos[neuronio_ganhador][a])); 
					}
				}
			}
		}
		
		if(necessidade_atualizar_pesos) {
			
			/* Reduzir a taxa de aprendizado, pode ser por meio de uma constante ou por meio de uma funcao */
			taxa_de_aprendizado = diminui_taxa_de_aprendizado(taxa_de_aprendizado); 
		}
		
		return contador_de_erros/((double)vetores_de_entrada.length);
	}
	
	/* Metodo que gera uma Matrix com as saidas da rede para uma sequencia de entradas */
	Matrix get_saidas(){
		double[] saidas_da_rede = new double[numero_de_instancias];
		
		/* Mesma sequencia de passos do get_saida(), porem nao possui as atualizacoes */
		int neuronio_ganhador = 0;
		distancia_euclidiana = new double[numero_neuronios_saida*numero_de_classes];	
		for(int k = 0; k < numero_de_instancias; k++){			
			neuronio_ganhador = 0;
			for(int i = 0; i< distancia_euclidiana.length; i++){
				distancia_euclidiana[i] = distancia_euclidiana(pesos[i] ,vetores_de_entrada[k]);
				
				/* A classe do neuronio ganhador eh colocada em uma matriz, sendo retornada */
				if(i!= 0){
					if(distancia_euclidiana[i]<distancia_euclidiana[neuronio_ganhador]){
						neuronio_ganhador = i;
						saidas_da_rede[k] = rotulo_pesos[i];
					}
				}
			}					
		}
			
		Matrix saidas = new Matrix(saidas_da_rede, numero_de_instancias);

		return saidas;
	}
	
	/* Diminui o numero de neuronios que cada classe possui, de acordo com um numero ideal de neuronios
	 * definido como parametro. Os neuronios retirados sao aqueles que sao menos ativados dentre os
	 * que sao daquela classe em questa passada como parametro. */
	public void corte_de_neuronios(int numero_neuronio_ideal, double classe, Matrix entradas_classe) {
		
		/*Obtem-se os indices dos pesos que sao referentes a classe que sera efetuado o corte */
		List<Integer> indice_neuronios_classe=new ArrayList<Integer>();
		for (int indice_peso = 0; indice_peso < pesos.length; indice_peso++) {
			if(rotulo_pesos[indice_peso]==classe) {
				indice_neuronios_classe.add(indice_peso);
			}
		}

		/* Armazena a quantidade de vezes em que cada um dos neuronios foi o vencedor */
		int[] vezes_ganhador=new int[indice_neuronios_classe.size()];
		
		/* Cada instancia, que tem como saida desejada igual a classe para corte, eh jogada na
		 * rede para calcular o neuronio ganhador*/
		for(int indice_instancia=0; indice_instancia<entradas_classe.getRowDimension(); indice_instancia++) {
			int neuronio_ganhador=0;
			
			/* Calcula o neuronio ganhador dessa instancia */
			double[] distancia_euclidiana=new double[indice_neuronios_classe.size()];
			for(int i=0; i< indice_neuronios_classe.size(); i++){ 
				int indice_peso=indice_neuronios_classe.get(i);
				double distancia = distancia_euclidiana(pesos[indice_peso] ,entradas_classe.getArray()[indice_instancia]);;
				distancia_euclidiana[i]=distancia;
				if(i!= 0){
					if(distancia_euclidiana[i]<distancia_euclidiana[neuronio_ganhador]){
						neuronio_ganhador = i;
					}
				}
			}
			
			/* Acrescentar o numero de vitorias do neuronio ganhador */
			vezes_ganhador[neuronio_ganhador]+=1;
		}
		List<Integer> indice_neuronios_classe_excluidos=new ArrayList<Integer>();
		
		/* Seleciona os indices na matriz de pesos, que serao excluidos */
		for (int indice_peso_i = 0; indice_peso_i < vezes_ganhador.length; indice_peso_i++) {
			int numero_neuronios_mais_vencedores=0; // numero de neuronio com mais vitorias do que esse neuronio
			for (int indice_peso_j = 0; indice_peso_j < vezes_ganhador.length; indice_peso_j++) {
				if(vezes_ganhador[indice_peso_i]<=vezes_ganhador[indice_peso_j]) {
					numero_neuronios_mais_vencedores+=1;
				}
			}
			
			/* Se o numero de neuronios mais vencedores que o neuronio iterado no primeiro laco
			 * for maior do que a quantidade ideal de neuronios, esse peso eh classificado como excluido */
			if(numero_neuronios_mais_vencedores > numero_neuronio_ideal ) {
				indice_neuronios_classe_excluidos.add( indice_neuronios_classe.get(indice_peso_i) );
				if(indice_neuronios_classe_excluidos.size()==(indice_neuronios_classe.size()-numero_neuronio_ideal)) {
					break;
				}
			}
		}
		
		/* Faz-se a atualizacao da matriz de pesos, deletando aqueles 
		 * neuronios que foram marcados para serem excluidos */
		int numero_linhas_novos_pesos=pesos.length - ( indice_neuronios_classe_excluidos.size());
		double[][] pesos_apos_corte=new double[numero_linhas_novos_pesos][pesos[0].length];
		double[] rotulos_apos_corte=new double[numero_linhas_novos_pesos];
		int linha_vazia_novos_pesos=0;
		for (int i = 0; i < pesos.length; i++) {
			
			/* Se o indice nao estiver na lista de excluidos */
			if( ! indice_neuronios_classe_excluidos.contains(i) ) {
				
				/* Adiciona esse pesos a matriz nova de pesos */
				rotulos_apos_corte[linha_vazia_novos_pesos]=rotulo_pesos[i];
				for (int j = 0; j < pesos[0].length; j++) {
					pesos_apos_corte[linha_vazia_novos_pesos][j]=pesos[i][j];
				}
				linha_vazia_novos_pesos+=1;
			}
		}
		pesos=pesos_apos_corte;
		rotulo_pesos=rotulos_apos_corte;
		numero_neuronios_saida=numero_neuronio_ideal;
	}
	
}
