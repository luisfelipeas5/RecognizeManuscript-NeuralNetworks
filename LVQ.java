/*
 *  Observacoes para o Senhor Marcelo
 *  Sobre os rotulos, decidi passar no construtor, então se vc reparar, eu mudei um pouco o construtor!
 */

/* 
 *  TODO Precisa ter o metodo de remover neuronios nao utilizados apos o treinamento e deve ser feito da seguinte maneira: se uma entrada for da 
 *  da classe x, podemos testar apenas com os neuronios da classe x (exemplo, se a entrada eh da classe 1, deve-se testar quais neuronios
 *  sao ativados da classe 1, apenas), entrada classe 2 -> neuronios da classe 2, e assim por diante. Podemos tentar fazer juntos, nao fiz
 *  pq nao sei como ele ira funcionar no treinamento exatamente
 * */

import Jama.Matrix;

import java.util.Arrays;
import java.util.Random;

public class LVQ extends Rede{
	
	double[][] vetores_de_entrada;	// dados de entrada
	double[] classe_alvo;			// vetor que guarda as classes das instancias (de entrada)
	int numero_de_instancias;		// vetores_de_entrada[i]
	int dimensao;					// da entrada
	int numero_de_classes;			// alvo, pode ser assumido como 10!!!
	
	double[][]pesos;				// aleatorio e sera passado pela classe treinamento
	double[] rotulo_pesos;			// classes dos vetores prototipos
	int numero_neuronios_saida;		// vetores prototipos para cada classe
	double taxa_de_aprendizado;		// definir
	int neuronio_ganhador;			// indice do vetor da classe ganhadora
	double[] distancia_euclidiana;	// conterah as distancias de uma entrada em relacao aos neuronios
	
	boolean necessiadade_atualizar_pesos=true;
	
	public LVQ(int numero_neuronios_classe, double taxa_de_aprendizado, double[] classes){	
		super(numero_neuronios_classe);
		this.numero_neuronios_saida = numero_neuronios_classe;
		this.taxa_de_aprendizado = taxa_de_aprendizado;
		this.numero_de_classes = classes.length;
		
		Arrays.sort(classes);
		this.rotulo_pesos=new double[numero_neuronios_classe*classes.length];
		int j=0;
		for(int i=0; i < classes.length;i++) {
			for (int k=0; k < numero_neuronios_classe; k++) {
				this.rotulo_pesos[j]=(double)classes[i];
				j++;
			}
		}
		//imprime_rotulos_dos_pesos();
		//System.exit(0);
	}
	
	/* As matrizes de entrada e saida desejada são transformadas em matrizes "normais" do java */
	void set_problema (Matrix entrada, Matrix saida_desejada){
		this.vetores_de_entrada = entrada.getArrayCopy();
		this.numero_de_instancias = entrada.getRowDimension();
		this.dimensao = entrada.getColumnDimension();
		this.classe_alvo = saida_desejada.getRowPackedCopy();		//ver certinho 
	}
	
	/* 
	 * Na LVQ, a matriz de pesos, que eh a matriz onde cada linha corresponde 
	 * a um vetor prototipo (neuronio) sera passado como parametro 
	 */
	void set_pesos (Matrix pesos_a, Matrix pesos_b){
		this.pesos = pesos_a.getArrayCopy();
	} 
	
	void set_modo_treinamento (int modo_treinamento){
		// Nao eh utilizado na LVQ
	} 
	
	/* Distancia Euclidiana sera utilizada no EP */
	public double distancia_euclidiana(double[] vetor1, double[] vetor2){
		double distancia =0;
		for(int j=0;j<dimensao;j++){
			distancia = distancia + Math.pow((vetor1[j] - vetor2[j]), 2);
		}
		return Math.sqrt(distancia);
	}
	
	/* A matriz de pesos tem o numero de linhas igual ao numero de neuronios de saida
	e numero de colunas igual ao numero de atributo, ou seja, a dimensao das entrasas */
	public void inicializa_pesos(){	
		pesos = new double[numero_neuronios_saida*numero_de_classes][dimensao];
		Random rd = new Random();
		for(int i = 0; i<numero_neuronios_saida*numero_de_classes; i++){
			for(int j = 0; j<dimensao; j++){
				pesos[i][j] = rd.nextDouble();
			} 	
		}
	}
	
	/* Imprime a matriz de pesos */
	void imprime_matriz_de_pesos(){
		for(int x=0; x < numero_neuronios_saida*numero_de_classes; x++){    
			for(int y=0; y < dimensao; y++){
                //System.out.print(pesos[x][y]+"\t");
            }
            //System.out.println();
        }
	} 
	
	/*void rotula_pesos(){ //Pode-se assumir dez classes
		rotulo_pesos = new double[numero_de_classes*numero_neuronios_saida];
	}*/
	
	/* Imprime a classe de cada vetor prototipo (neuronio) */
	
	void imprime_rotulos_dos_pesos(){
		for (int y = 0; y < rotulo_pesos.length; y++){
			System.out.println(rotulo_pesos[y]);
		}	
	}
	
	/* A constante pode ser trocada, porem como visto em aula, 0.9 eh um bom numero
	 * Funcoes tambem podem ser utilizadas para o decrescimento da taxa */
	public double diminui_taxa_de_aprendizado(double taxa_de_aprendizado_atual){
		double taxa_atualizada;
		taxa_atualizada = taxa_de_aprendizado_atual*0.9;
		return taxa_atualizada;
	}

	/* Retornar o erro de um unica epoca, ou seja, a quantidade de vezes que o neuronio ganhador nao era da mesma classe que a instancia
	 * Aqui eh feito a primeira epoca da lvq, no treinamento eh adicionado apenas um laco para a quantidade de epocas desejadas */
	double get_erro(){	
	
		double contador_de_erros = 0;
		distancia_euclidiana = new double[numero_neuronios_saida*numero_de_classes];//matriz que guardarah as distancias de uma instancia em relacao a cada vetor prototipo

		//imprime_matriz_de_pesos();
		//System.out.println("");
		//System.out.println("Epoca 0");
		//System.out.println("");
			
		/* Passo 1 - Para cada vetor de entrada, executa os passos 3 e 4 */
			
		for(int k = 0; k < numero_de_instancias; k++){	
		
			/* Passo 2 - Encontrar o vetor (neuronio) de saida tal que a distancia seja minima */
			
			neuronio_ganhador = 0;
			for(int i = 0; i< numero_neuronios_saida*numero_de_classes; i++){
				distancia_euclidiana[i] = distancia_euclidiana(pesos[i] ,vetores_de_entrada[k]);
				if(i!= 0){
					if(distancia_euclidiana[i]<distancia_euclidiana[neuronio_ganhador]){
						neuronio_ganhador = i;
					}
				}
				//System.out.println(distancia_euclidiana[i] + "	neuronio ganhador:" + neuronio_ganhador);
			}
		
			/*
			 * Se os pesos devem ser atualizados (conjunto de entradas eh o conjunto de treinamento)
			 *  (nao se deve atualizar os peses quando temos o conjunto de validacao como conjunto de entradas):
			 */
			if(necessiadade_atualizar_pesos) {
				/* Passo 3 - Alterando os pesos analisando a classe definida. Sao aplicados as regras de aprendizado. A soma ocorre quando 
				a classificacao sugerida pela LVQ eh correta (o vetor resultante da operação está situado entre o vetor protótipo e o vetor
				de dados, ou seja, houve a movimentação do vetor protótipo na direção do dado). Já a subtracao ocorre quando a classificacao
				sugerida eh incorreta (tendo efeito contrario) e o contador de erros eh incrementado */
				
				//imprime_rotulos_dos_pesos();
				//System.out.println("k="+k+" classe_alvo.length="+classe_alvo.length);
				//System.out.println("neuronio_ganhador="+neuronio_ganhador+ "rotulo_pesos="+rotulo_pesos.length);
				
				if(classe_alvo[k] == rotulo_pesos[neuronio_ganhador]){	// Preciso colocar os rotulos dos pesos
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
		if(necessiadade_atualizar_pesos) {
			/* Reduzir a taxa de aprendizado, pode ser por meio de uma constante (sendo linear) ou por meio de uma funcao */
			taxa_de_aprendizado = diminui_taxa_de_aprendizado(taxa_de_aprendizado);
			//System.out.println("Taxa Aprendizado="+taxa_de_aprendizado); 
		}
		return contador_de_erros;
	}
	
	/* Metodo que gera uma Matrix com as saidas da rede para uma sequencia de entradas */
	Matrix get_saidas(){
		double[] saidas_da_rede = new double[numero_de_instancias];
		
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
	
}
