import java.util.Random;
import Jama.Matrix;

public class Treinamento {
	public void treina_mlp(Matrix entrada, Matrix saida_desejada, int epocas, Rede rede) {
		//Para a primeira epoca, os pesos devem ser gerados randomicamente
		
		int linhas_pesos_a = rede.numero_neuronios_escondidos;
		int colunas_pesos_a = entrada.getColumnDimension();
		double[][] array_pesos_a=new double[ linhas_pesos_a ][ colunas_pesos_a ];
		this.gera_pesos_aleatorios(array_pesos_a);
		Matrix pesos_a=new Matrix( array_pesos_a);
		
		int linhas_pesos_b = 1; //numero de neuronios de saida: nessa MLP em especifico, um unico neuronio de saida
		int colunas_pesos_b = rede.numero_neuronios_escondidos+1; //Mais o bias
		double[][] array_pesos_b=new double[ linhas_pesos_b ][ colunas_pesos_b ];
		this.gera_pesos_aleatorios(array_pesos_b);
		Matrix pesos_b=new Matrix( array_pesos_b);
		
		Matrix saidas=new Matrix(1, rede.numero_neuronios_saida);
		for (int i = 0; i < entrada.getRowDimension(); i++) {
			Matrix saida=rede.calcula_saida(entrada, saida_desejada, pesos_a, pesos_b);
			saidas.setMatrix(new int [] {i}, new int[] {1}, saida );
		}
		
		double erro_total=this.calcula_erro_total(saida_desejada, saidas);	
		
		double erro_desejado=0.05;//Deifinir qual seria uma taxa de erro razoavel
		for( int epoca_atual=1; epoca_atual<epocas && erro_total>erro_desejado; epoca_atual++ ) {
			/*Caso a atualizacao de erro seja a batelada
			//calcular Gradiente
			//Matrix gradiente=this.calculaGradiente();
			//calcular Passo
			//double alpha=this.calculaPasso();
			//Atualizar pesos
			//this.atualizaPesos(pesosA,alpha);
			//this.atualizaPesos(pesosB,alpha);
			*/
			for (int i = 0; i < entrada.getRowDimension(); i++) {
				Matrix saida=rede.calcula_saida(entrada, saida_desejada, pesos_a, pesos_b);
				saidas.setMatrix(new int [] {i}, new int[] {1}, saida );
			}
			
			double erro_total_antigo=erro_total;
			erro_total=this.calcula_erro_total(saida_desejada, saidas);
			while (erro_total>erro_total_antigo) {
				//TODO: calcular o erro_total ajustando o passo
				erro_total=erro_total/2;
			}
		}
		
		int casas_decimais=3;
		System.out.println("Pesos A");
		pesos_a.print(colunas_pesos_a, casas_decimais);
		System.out.println("Pesos B");
		pesos_b.print(colunas_pesos_b, casas_decimais);
		System.out.println("Saida");
		saidas.print(saidas.getColumnDimension(), casas_decimais);
		System.out.println("Saida Desejada");
		saida_desejada.print(saida_desejada.getColumnDimension(), casas_decimais);
		
	}
	
	
	
	private double calcula_erro_total(Matrix saida_desejada, Matrix saida) {
		/*
		*calcular o erro de cada instância processada pela rede e depois
		*calcular o erro quadrado para determinar o erro total no final da epoca (a primeira epoca no caso)
		*/
		//erroQuadrado = ( saidaDesejada - saida) ^ 2		
		Matrix erro_quadrado=saida_desejada.minus(saida);//Erros estao sem a operacao de potencia
		//Elevar os erros ao quadrado
		double erro_total=0;
		for(int i=0; i<erro_quadrado.getRowDimension(); i++) {
			for(int j=0; j<erro_quadrado.getColumnDimension(); j++) {
				double elemento_antigo=erro_quadrado.get(i, j);
				erro_quadrado.set(i,	j, elemento_antigo*elemento_antigo);
				erro_total=erro_total+(elemento_antigo*elemento_antigo);
			}
		}
		int casas_decimais=3;
		System.out.println("Erro Quadrado");
		erro_quadrado.print(erro_quadrado.getColumnDimension(), casas_decimais);
		return erro_total;
	}



	private Matrix calcula_saida(Matrix entrada, Matrix pesos_a, Matrix pesos_b,int numero_neuronios_escondidos) {
		//Obter a saida para cada uma das instancias na matriz de entrada
		double[][] array_saida = new double[entrada.getRowDimension()][1];
		/*
		for(int i=0; i<saida.length; i++){
			arraySaida[i][0]=MLP(entrada[i], pesosA, pesosB, numeroNeuroniosEscondidos);
		}
		*/
		
		//Por enquanto que nao temos a rede MLP implementada, geramos valores aleatorio para as saidas
		//this.geraPesosAleatorios(arraySaida);
		Matrix saida=new Matrix(array_saida);
		return saida;
	}



	public void atualiza_pesos(Matrix pesos, double alpha) {
		//wNew=wOld-alpha*gradiente
		for(int i=0; i<pesos.getRowDimension(); i++) {
			for(int j=0; j<pesos.getColumnDimension(); j++) {
				double pesoAntigo = pesos.get(i, j);
				double gradienteNoPonto=this.calcula_gradiente_no_ponto(pesoAntigo);
				double pesoAtualizado=pesoAntigo-gradienteNoPonto*alpha;
				pesos.set(i, j, pesoAtualizado);
			}
		}
	}

	private double calcula_gradiente_no_ponto(double x) {
		return 1;
	}

	public double calcula_passo() {	
		//TODO Algortitmo da Bissecao
		return 0.1;
	}

	/*
	 * Esse metodo, dado uma matriz pesos de dimensoes quaisquer, preenche pesos com valores aleatorios
	 * entre 0.0 e 1.0 
	 */
	public void gera_pesos_aleatorios(double[][] pesos) {
		Random random=new Random();
		for(int i=0; i<pesos.length; i++) {
			for(int j=0; j<pesos[i].length; j++) {
				pesos[i][j]=random.nextDouble();
			}
		}
	}
}
