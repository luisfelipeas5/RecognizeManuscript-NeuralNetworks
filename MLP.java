//Pacote Jama: 
import Jama.Matrix;
//Estruturas de dados: 
import java.util.List; 
import java.util.LinkedList; 
//Instrumentos matematicos: 
import java.lang.Math; 
import java.lang.Double; 
import java.lang.Integer; 
//Classe que gera numeros aleatorios: 
import java.util.Random;
//Excecoes: 
import java.lang.IllegalArgumentException;
import java.lang.ArrayIndexOutOfBoundsException; 

public class MLP {
	/*Durante a codificacao deste exercicio, supoe-se que o numero de neuronios eh o mesmo em todas 
	as camadas escondidas (cuja quantidade, embora possa variar, foi fixada como igual a um) */
	int neuronios_por_camada; 
	//matrizes que representam o conjunto de dados de treinamento, validacao e teste.
	Matrix entrada; 
	Matrix saida_desejada; 
	int numero_entradas; //numero de entradas da rede
	int numero_saidas; //numero de saidas da rede
	double taxa_aprendizado; //taxa de aprendizado
	Matrix[] matrizes_pesos; /*Vetor que contem as matrizes de pesos que compoem a rede neural. Visto 
	que ha apenas uma camada intermediaria, este vetor possui tamanho 2*/ 
	//Os atributos de classe a seguir merecem uma atencao especial 
	/*Durante o processo "propagation", primeiro multiplicamos a matriz de entrada pela primeira matriz
	de pesos. Ao resultado, aplicamos uma funcao de ativacao (no caso deste ep, eh a funcao sigmoide).
	A matriz existente antes da aplicacao da funcao foi colocada na posicao 0 do vetor semi_results
	enquanto que a matriz resultante desta aplicacao foi salva na posicao 1 do vetor saidas_rede. Alem 
	disso, a matriz correspondente a saidas_rede[0] eh multiplicada pela segunda matriz de pesos e uma 
	nova matriz eh obtida. Essa nova matriz eh colocada na posicao 1 de semi_results. Ao aplicar uma
	nova funcao de ativacao nessa matriz (que poderia ser linear, mas, em vez disso, foi utilizada uma 
	nova sigmoide), obtem-se a matriz correspondente a posicao 1 do vetor saidas_rede*/
	Matrix[] semi_results; 
	Matrix[] saidas_rede; 
	Matrix erro; //Matriz resultante da subtracao da matriz de saida pela de saida desejada
	List<Double> En; /*Lista que conterah o erro quadratico para cada saida. Nesse caso, particularmente,
	teremos uma lista de um unico elemento*/  
	
	/*Construtor da classe MLP: recebe como parametros o numero de neuronios existentes na camada escondida,
	a taxa de aprendizado alfa e as matrizes de entrada e de saida esperada*/
	public MLP (int n_neuronios, double alfa, Matrix entrada, Matrix saida) {
		this.neuronios_por_camada = n_neuronios; 
		this.entrada = entrada; 
		this.saida_desejada = saida;
		double[][] s = saida.getArrayCopy(); 
		double[][] e = entrada.getArrayCopy(); 
		this.numero_entradas = e[0].length; 
		this.numero_saidas = s[0].length; 
		matrizes_pesos = new Matrix[2];
		this.taxa_aprendizado = alfa; 
	}
	
	/*Infelizmente, no pacote Jama (que serah enviado junto com as classes deste EP), nao foi encontrado um metodo
	que realize o produto entre duas matrizes. Portanto, cada elemento (i,j) da matriz gerada por este metodo serah 
	o produto interno entre o vetor correspondente a linha i da primeira matriz e o vetor correspondente a coluna j 
	da segunda matriz. Se o numero de colunas da primeira matriz for diferente do numero de linhas da segunda matriz, 
	diz-se que as matrizes sao incompativeis e uma excecao serah lancada*/
	public Matrix produto (Matrix A, Matrix B) {
		double[][] a = A.getArrayCopy(); 
		double[][] b = B.getArrayCopy(); 
		if (a[0].length != b.length) {
			throw new IllegalArgumentException("No produto A por B, o numero de colunas de A eh diferente do numero de colunas de B."); 
		}
		else {
			double[][] c = new double[a.length][b[0].length]; 
			for (int i = 0; i < c.length; i++) {
				for (int j = 0; j < c[0].length; j++) {
					double aux = 0; 
					for(int k = 0; k < a[0].length; k++) {
						aux = aux + a[i][k]*b[k][j]; 
					}
					c[i][j] = aux; 
				}
			}
			return new Matrix(c); 
		}
	}
	
	/*O metodo a seguir inicializa a rede criando duas matrizes de pesos aleatorias. Tal como foi explicado em aula, a 
	primeira matriz tem o numero de linhas igual ao numero de neuronios da camada intermediaria e o numero de colunas 
	igual ao numero de entradas (mais um, para representar o bias). A segunda matriz, por sua vez, tem como numero de 
	linhas o numero de neuronios na camada intermediaria e como numero de colunas o numero de saidas (mais um, para 
	indicar o bias).*/
	public void inicializa_rede() {
		double[][] matriz_pesos = new double[neuronios_por_camada][numero_entradas+1]; 
		for (int i = 0; i < matriz_pesos.length; i++) {
			for (int j = 0; j < matriz_pesos[0].length; j++) {
				matriz_pesos[i][j] = Math.random() - 0.5; 
			}
		}
		matrizes_pesos[0] = new Matrix(matriz_pesos);
		matriz_pesos = new double[numero_saidas][neuronios_por_camada+1];
		for (int i = 0; i < matriz_pesos.length; i++) {
			for (int j = 0; j < matriz_pesos[0].length; j++) {
				matriz_pesos[i][j] = Math.random() - 0.5; 
			} 
		}
		matrizes_pesos[1] = new Matrix(matriz_pesos); 	
	}
	
	//Funcao de ativacao (logistica)
	public double sigmoide(double x) {
		return 1.0/(1.0+Math.exp((-1.0)*x)); 
	}
	
	/*Metodo que aplica a funcao de ativacao a cada elemento de uma matriz */
	public Matrix f(Matrix x) {
		double[][] x_aux = x.getArrayCopy(); 
		double[][] x_apf = new double[x_aux.length][x_aux[0].length]; 
		for (int i = 0; i < x_apf.length; i++) {
			for (int j = 0; j < x_apf[0].length; j++) {
				x_apf[i][j] = sigmoide(x_aux[i][j]); 
			}
		}
		return new Matrix(x_apf); 
	}
	
	//Derivada da funcao de ativacao
	public double sigmoide_linha (double x) { 
		return sigmoide(x)*(1.0-sigmoide(x)); 
	}
	
	/*Metodo propagation: consiste em calcular a saida da rede, que serah comparada posteriormente
	a saida esperada ja que se trata de um projeto supervisionado. Inicialmente, a matriz de entrada 
	eh copiada para uma matriz auxiliar, digamos X. P eh uma matriz que conterah o produto de duas matrizes. 
	Alem disso, os vetores semi_results e saidas_rede sao instanciados como tendo tamanho igual a 2 (como ja 
	foi explicado anteriormente). Dentro do ciclo for, eh criada uma nova matriz de numero de linhas igual ao 
	de X e o numero de colunas igual ao de X acrescido de uma unidade (para representar o bias). As primeiras 
	X[0].length colunas serao prenchidas com os valores originais. A ultima coluna, contudo, terah apenas 
	valores 1. Esse procedimento eh feito para nao haver problemas ao multiplicar a matriz de entrada com a 
	primeira matriz de pesos. Assim, p recebe o produto de X pela transposta da matriz de pesos e, como dito
	anteriormente, seu valor eh salvo em semi_results. Apos aplicarmos uma funcao de ativacao, a nova matriz eh 
	salva em saidas_rede. Um procedimento semelhante eh realizado em seguida, considerando a segunda matriz de 
	pesos e a matriz de "saida" da primeira parte da rede como a nova matriz de entrada. No fim do ciclo for,
	teremos uma matriz de saida da rede, onde o numero de linhas eh igual ao numero de instancias e o de colunas 
	eh 1, visto que soh ha uma saida na rede. Ao subtrairmos a matriz de saida pela de saida esperada, obtem-se 
	a matriz de erro. Alem disso, podemos elevar cada erro ao quadrado e assim teremos um valor de erro quadratico 
	para cada instancia. A soma de todos os erros quadradicos dividida pelo numero de instancias serah o erro
	quadratico medio, que sera retornado pelo metodo. */
	public double propagation() {
		Matrix entrada_aux = entrada; 
		try {
			Matrix p = null; 
			semi_results = new Matrix[2]; 
			saidas_rede = new Matrix[2]; 
			for (int k = 0; k < 2; k++) {
				double[][] aux = entrada_aux.getArrayCopy(); 
				double[][] matriz_entrada = new double[aux.length][aux[0].length+1];
				for (int i = 0; i < matriz_entrada.length; i++) {
					for (int j = 0; j < matriz_entrada[0].length-1; j++) {
						matriz_entrada[i][j] = aux[i][j];
					}
					matriz_entrada[i][matriz_entrada[0].length-1] = 1.0; 			
				}
				entrada_aux = new Matrix(matriz_entrada); 
				p = produto(entrada_aux,matrizes_pesos[k].transpose());
				semi_results[k] = p; 
				entrada_aux = f(p); 
				saidas_rede[k] = entrada_aux; 
			}
			this.En = new LinkedList<Double>(); 
			erro = saidas_rede[1].minus(saida_desejada); 
			double[][] e = erro.getArrayCopy(); 
			double Et = 0.0; 
			for (int i = 0; i < e.length; i++) {
				En.add(Math.pow(e[i][0],2)/2.0); 
				Et = Et + En.get(i); 
			}
			double Em = Et/e.length;
			return Em; 
		}
		catch (IllegalArgumentException i) {
			System.out.println (i.getMessage()); 
			return 0.0; 
		}
	}
	
	/* Parametros: 
	--> n: indice referente a matriz de pesos localizada na posicao n do vetor
	--> i: linha da matriz onde se encontra o peso a ser melhorado
	--> j: coluna da matriz onde se encontra o peso a ser melhorado 
	--> inst: instancia sobre a qual a melhoria serah calculada 
	O funcionamento deste metodo pode ser dividido em dois grandes casos: 
	Quando n = 1, estamos nos referindo a segunda matriz de pesos. Nesse caso,
	para calcular o quanto devemos adicionar a um peso w(i,j) especifico, usamos 
	a regra delta. Ou seja, precisa-se multiplicar quatro fatores: 
	--> taxa de aprendizado; 
	--> o erro resultante da saida i para certa instancia
	--> o valor da derivada da funcao de ativacao calculada sobre o valor da saida 
	da rede para o neuronio de saida i e uma instancia especifica (antes de aplicar a funcao 
	de ativacao)
	--> o valor da entrada desta parte da rede (ou seja, a saida j da primeira parte da 
	rede. 
	Quando n = 0, estamos nos referindo a primeira matriz de pesos. Nesse caso, 
	precisa-se primeiramente retropropagar o erro da saida da rede para a "saida"
	da primeira parte da rede. Isso eh possivel por meio da soma dos valores da 
	propagacao de erro de cada saida da rede. Cada propagacao eh feita atraves da 
	multiplicacao dos seguintes termos: 
	--> o erro resultante de uma saida k para certa instancia (onde k eh um valor que serah
	incrementado e o valor maximo de k eh o numero de saidas menos 1)
	--> o valor da derivada da funcao de ativacao calculada sobre o valor da saida da rede para o neuronio
	de saida k e uma instancia especifica (antes de aplicar a funcao)
	--> o peso da "transicao" que ocorre entre o neuronio intermediario i e o neuronio de saida k (existente
	na segunda matriz de pesos). 
	O valor resultante do produto descrito acima eh multiplicado pelo produto de tres outros fatores: 
	--> taxa de aprendizado
	--> o valor da derivada da funcao de ativacao calculado sobre a saida i da primeira parte da rede (antes de
	aplicar a funcao) 
	--> o valor da entrada j em uma instancia especifica
	Observacao: Este metodo mostra a importancia dos vetores de objetos Matriz semi_results e saidas_redes jah discutidos
	anteriormente
	*/
	public double calcula_melhoria (int n, int i, int j, int inst) {
		if (n == 1) {
			double ei_n = 0.0; //erro simples para o neuronio i no instante n
			double[][] erros = erro.getArrayCopy(); 
			ei_n = erros[inst][i]; 
			double fl_vin = 0.0; 
			double[][] saida_rede_af = semi_results[1].getArrayCopy(); 
			fl_vin = sigmoide_linha(saida_rede_af[inst][i]); 
			double yj_n = 0.0;
			double[][] pseudo_entrada = saidas_rede[0].getArrayCopy(); 
			double[][] pseudo_entrada2 = new double[pseudo_entrada.length][pseudo_entrada[0].length+1]; 
			for (int m = 0; m < pseudo_entrada2.length; m++) {
				for (int o = 0; o < pseudo_entrada2[0].length -1; o++) {
					pseudo_entrada2[m][o] = pseudo_entrada[m][o]; 
				}
				pseudo_entrada2[m][pseudo_entrada2[0].length - 1] = 1.0; 
			}
			yj_n = pseudo_entrada2[inst][j]; 
			return taxa_aprendizado*ei_n*fl_vin*yj_n; 	
		}
		else if (n == 0) {
			double ei_n = 0.0; 
			double[][] saida_rede_af; 
			for (int k = 0; k < numero_saidas; k++) {
				double[][] erros = erro.getArrayCopy(); 
				double ek_n = erros[inst][k]; 
				saida_rede_af = semi_results[1].getArrayCopy(); 
				double fl_vkn = sigmoide_linha(saida_rede_af[inst][k]);
				double[][] seg_mat_pesos = matrizes_pesos[1].getArrayCopy(); 
				ei_n = ei_n + seg_mat_pesos[k][i]*ek_n*fl_vkn; 				
			}
			saida_rede_af = semi_results[0].getArrayCopy(); 
			double fl_vin = sigmoide_linha(saida_rede_af[inst][i]); 
			double[][] ent = entrada.getArrayCopy();
			double[][] ent2 = new double[ent.length][ent[0].length+1]; 
			for (int m = 0; m < ent2.length; m++) {
				for (int o = 0; o < ent2[0].length -1; o++) {
					ent2[m][o] = ent[m][o]; 
				}
				ent2[i][ent2[0].length - 1] = 1.0; 
			}
			double xj_n = ent2[inst][j];
			return taxa_aprendizado*ei_n*fl_vin*xj_n; 
		}
		else {
			throw new IllegalArgumentException("Matriz de pesos inexistente"); 
		}
	}
	
	/*O metodo treina_matriz_pesos aplica a funcao calcula_melhoria (explicada anteriormente) a cada elemento da matriz
	indicada pelo inteiro indice_matriz apos a execucao do metodo propagation() em uma dada instancia. Assim, diz-se que 
	trata-se de uma aplicacao do metodo padrao a padrao.*/
	public void treina_matriz_pesos (int indice_matriz, int instancia) {
		if (indice_matriz < matrizes_pesos.length && indice_matriz >= 0) {
			try {
				double[][] mat = matrizes_pesos[indice_matriz].getArrayCopy();
				for (int i = 0; i < mat.length; i++) {
					for (int j = 0; j < mat[0].length; j++) {
						mat[i][j] = mat[i][j] + calcula_melhoria(indice_matriz, i, j, instancia); 
					}
				}				
			}
			catch (ArrayIndexOutOfBoundsException a) {
				System.out.println ("Erro ao acessar um campo inexistente de uma matriz. Por favor, verifique o arquivo MLP.java"); 
			}
		}
		else {
			throw new IllegalArgumentException("Matriz de pesos inexistente");
		}
	}
	
	/*O metodo backpropagation serve para treinar as duas matrizes de pesos uma vez para cada instancia do arquivo de entrada. A
	instancia a ser treinada eh escolhida aleatoriamente. O vetor indices_jah_usados armazena as instancias que jah foram treinadas*/
	public void backpropagation() {
		int i;
		Random r = new Random(); 
		List<Integer> indices_jah_usados = new LinkedList<Integer>(); 
		while (indices_jah_usados.size() != entrada.getArrayCopy().length) {
			i = r.nextInt(entrada.getArrayCopy().length);
			if (!indices_jah_usados.contains(new Integer(i))) {
				indices_jah_usados.add(new Integer(i)); 
				treina_matriz_pesos(1, i); 
				treina_matriz_pesos(0, i);
			}
		}
	}	
}
