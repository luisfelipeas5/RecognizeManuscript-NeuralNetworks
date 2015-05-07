import java.util.Random;
import Jama.Matrix;


public class Treinamento {
	public void treinaMLP(double[][] arrayEntrada, double[][] arraySaidaDesejada,
			int numeroNeuroniosEscondidos, int epocas) {
		
		//Para a primeira epoca, os pesos devem ser gerados randomicamente
		
		int linhasPesosA = numeroNeuroniosEscondidos;
		int colunasPesosA = arrayEntrada[0].length;
		double[][] arrayPesosA=new double[ linhasPesosA ][ colunasPesosA ];
		this.geraPesosAleatorios(arrayPesosA);
		Matrix pesosA=new Matrix( arrayPesosA);
		
		int linhasPesosB = 1; //numero de neuronios de saida: nessa MLP em especifico, um unico neuronio de saida
		int colunasPesosB = numeroNeuroniosEscondidos+1; //Mais o bias
		double[][] arrayPesosB=new double[ linhasPesosB ][ colunasPesosB ];
		this.geraPesosAleatorios(arrayPesosB);
		Matrix pesosB=new Matrix( arrayPesosB);
		
		//Obter a saida para cada uma das instancias na matriz de entrada
		double[][] arraySaida = new double[arrayEntrada.length][1];
		/*
		for(int i=0; i<saida.length; i++){
			arraySaida[i][0]=MLP(entrada[i], pesosA, pesosB, numeroNeuroniosEscondidos);
		}
		*/
		//Por enquanto que nao temos a rede MLP implementada, geramos valores aleatorio para as saidas
		this.geraPesosAleatorios(arraySaida);
		Matrix saida=new Matrix(arraySaida);
		
		/*
		*calcular o erro de cada instância processada pela rede e depois
		*calcular o erro quadrado para determinar o erro total no final da epoca (a primeira epoca no caso)
		*/
		//erroQuadrado = ( saidaDesejada - saida) ^ 2		
		Matrix saidaDesejada=new Matrix(arraySaidaDesejada);
		Matrix erroQuadrado=saidaDesejada.minus(saida);//Erros estao sem a operacao de potencia
		//Elevar os erros ao quadrado
		double erroTotal=0;
		for(int i=0; i<erroQuadrado.getRowDimension(); i++) {
			for(int j=0; j<erroQuadrado.getColumnDimension(); j++) {
				double elementoAntigo=erroQuadrado.get(i, j);
				erroQuadrado.set(i,	j, elementoAntigo*elementoAntigo);
				erroTotal=erroTotal+(elementoAntigo*elementoAntigo);
			}
		}
		
		double erroDesejado=0.05;//Deifinir qual seria uma taxa de erro razoavel
		for( int epocaAtual=1; epocaAtual<epocas && erroTotal>erroDesejado; epocaAtual++ ) {
			
		}
		
		int casasDecimais=3;
		System.out.println("Pesos A");
		pesosA.print(colunasPesosA, casasDecimais);
		System.out.println("Pesos B");
		pesosB.print(colunasPesosB, casasDecimais);
		System.out.println("Saida");
		saida.print(saida.getColumnDimension(), casasDecimais);
		System.out.println("Saida Desejada");
		saidaDesejada.print(saidaDesejada.getColumnDimension(), casasDecimais);
		System.out.println("Erro Quadrado");
		erroQuadrado.print(erroQuadrado.getColumnDimension(), casasDecimais);
		
	}

	/*
	 * Esse metodo, dado uma matriz pesos de dimensoes quaisquer, preenche pesos com valores aleatorios
	 * entre 0.0 e 1.0 
	 */
	public void geraPesosAleatorios(double[][] pesos) {
		Random random=new Random();
		for(int i=0; i<pesos.length; i++) {
			for(int j=0; j<pesos[i].length; j++) {
				pesos[i][j]=random.nextDouble();
			}
		}
	}
}
