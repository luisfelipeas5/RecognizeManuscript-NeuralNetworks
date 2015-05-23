import Jama.Matrix;
import edu.umbc.cs.maple.utils.*;
import optimization.*;
// http://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/edu/umbc/cs/maple/utils/JamaUtils.html
// http://www.seas.upenn.edu/~eeaton/software.html
public class Pre_Processamento{
	
	public static Matrix max(Matrix dados){
		Matrix resultado = new Matrix(dados.getRowDimension(),dados.getColumnDimension());
	
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			for (int linha = 0; linha < dados.getRowDimension(); linha++) {
				if(JamaUtils.getMax(JamaUtils.getcol(dados, coluna))!=0){
					resultado.set(linha, coluna, dados.get(linha, coluna)/JamaUtils.getMax(JamaUtils.getcol(dados, coluna)));
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		return resultado;
	}
	
	public static Matrix minmax(Matrix dados){
		double[] intervCol = new double[dados.getColumnDimension()];
		Matrix resultado = new Matrix(dados.getRowDimension(),dados.getColumnDimension());
		
		intervCol = calculaIntervalo(dados);
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			for (int linha = 0; linha < dados.getRowDimension(); linha++) {
				if(intervCol[coluna]!=0){
					resultado.set(linha, coluna, ((dados.get(linha, coluna) - JamaUtils.getMin(JamaUtils.getcol(dados, coluna)))/intervCol[coluna])); 
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		
		return resultado;
	}
	
	public static Matrix sigmoidal(Matrix dados){
		double[] desCol = new double[dados.getColumnDimension()];
		double[] medCol = new double[dados.getColumnDimension()];
		Matrix resultado = new Matrix(dados.getRowDimension(),dados.getColumnDimension());
		
		medCol = calculaMedia(dados);
		desCol = calculaVariancia(dados);
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			for (int linha = 0; linha < dados.getRowDimension(); linha++) {
				if(desCol[coluna]!=0){
					resultado.set(linha, coluna, 1/(1+Math.exp(-(dados.get(linha, coluna) - medCol[coluna])/desCol[coluna])));
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		
		return resultado;
	}
	
	public static Matrix zscore(Matrix dados){
		
		double[] medCol = new double[dados.getColumnDimension()];
		double[] varCol = new double[dados.getColumnDimension()];
		Matrix resultado = new Matrix(dados.getRowDimension(),dados.getColumnDimension());
		
		medCol = calculaMedia(dados);
		varCol = calculaVariancia(dados);
		
		for(int coluna = 0; coluna < dados.getColumnDimension(); coluna++){
			for(int linha = 0; linha < dados.getRowDimension(); linha++){
				if(varCol[coluna]!=0){
					resultado.set(linha, coluna, (dados.get(linha, coluna)-medCol[coluna])/Math.sqrt(varCol[coluna]));
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		
		return resultado;
	}
	
	public static Matrix removeZeros(Matrix dados, int porcentagem){
		/* Corte eh calculado a partir da porcentagem estipulada durante a chamada do metodo
		Ex: Se deseja-se eliminar todas as colunas compostas 100% de zeros, basta passar
		int porcentagem = 100. Com isso, corte = numero de linhas das colunas. */
		
		/* O codigo abaixo serve para preencher a matrix aux apenas com as colunas que
		possuem desvio padrao maior que o limite desejado.
		As colunas com numero de zeros maior que o desejado sao "puladas", ou seja
		nao sao copiadas para a matriz aux. A matriz aux possui as mesmas dimensoes
		da matriz de dados fornecida. Isso pode ser constatado em sua inicializacao.
		 
		Como algumas colunas serao "puladas" (seus valores nao serao inseridos),
		as suas ultimas X colunas ficarao preenchidas com zeros. Saberemos quais serao
		as ultimas colunas a ser removidas atraves do contatos removeColunas 
		(X = removeColunas). Ao fim do codigo teremos uma matriz de dimensao 
		m por (n - removeColunas), onde m e n sao os numeros de linhas e colunas 
		da matriz de dados passada como parametro da funcao. */
		
		int corte = (porcentagem/100)*dados.getRowDimension();
		int contaZeros = 0;
		int removeColunas = 0;
		Matrix aux = new Matrix(dados.getColumnDimension(),dados.getRowDimension());
		Matrix resultado;
		
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			for (int linha = 0; linha < dados.getRowDimension(); linha++) {
				if(dados.get(linha, coluna) == 0){
					contaZeros++;
				}
			}
			// Se a quantidade de zeros for menor que o corte, adiciona essa coluna
			if(contaZeros < corte){
				for (int linha = 0; linha < dados.getRowDimension(); linha++) {
						aux.set(linha, coluna, dados.get(linha, coluna));
				}
			}else{
				removeColunas++;
			}
			contaZeros = 0;
		}
		
		/* Cria uma matriz de dimensoes reduzidas, ja que colunas serao removidas as ultimas
		X colunas. */
		resultado = new Matrix(dados.getRowDimension(), dados.getColumnDimension()-removeColunas);
		
		/* Pega a submatriz de linhas 1 a m e colunas 1 a (n - removeColunas), sendo m e n
		a dimensao das linhas e colunas da matriz de dados fornecida como parametro. */
		resultado.getMatrix(1, dados.getRowDimension(), 1, dados.getRowDimension()-removeColunas);
		
		return resultado;
	}
	
	public static Matrix removeDesvioBaixo(Matrix dados, int limiar){
		/* Corte eh calculado a partir da porcentagem estipulada durante a chamada do metodo
		Ex: Se deseja-se eliminar todas as colunas compostas 100% de zeros, basta passar
		int porcentagem = 100. Com isso, corte = numero de linhas das colunas. */
		
		/* O codigo abaixo serve para preencher a matrix aux apenas com as colunas que
		possuem desvio padrao maior que o limite desejado.
		As colunas com desvio padrao maior que o limite sao "puladas", ou seja
		nao sao copiadas para a matriz aux. A matriz aux possui as mesmas dimensoes
		da matriz de dados fornecida. Isso pode ser constatado em sua inicializacao.
		 
		Como algumas colunas serao "puladas" (seus valores nao serao inseridos),
		as suas ultimas X colunas ficarao preenchidas com zeros. Saberemos quais serao
		as ultimas colunas a ser removidas atraves do contatos removeColunas 
		(X = removeColunas). Ao fim do codigo teremos uma matriz de dimensao 
		m por (n - removeColunas), onde m e n sao os numeros de linhas e colunas 
		da matriz de dados passada como parametro da funcao. */
				
		double[] varCol = new double[dados.getColumnDimension()];
		// Inicializa aux como uma matriz de zeros de m por n colunas
		Matrix aux = new Matrix(dados.getColumnDimension(),dados.getRowDimension());
		Matrix resultado;
		int removeColunas = 0;
		
		varCol = calculaVariancia(dados);
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			for (int linha = 0; linha < dados.getRowDimension(); linha++) {
				// Se o desvio padrao for maior que o limiar desejado, todas as linhas
				// dacoluna sao copiadas.
				if(Math.sqrt(varCol[coluna]) > limiar){
					aux.set(linha, coluna, dados.get(linha, coluna));
				}else{
					// Caso contrario, indica-se que mas uma coluna devera ser removida
					removeColunas++;
				}
			}
		}
		
		/* Cria uma matriz de dimensoes reduzidas, ja que colunas serao removidas as ultimas
		X colunas. */
		resultado = new Matrix(dados.getRowDimension(), dados.getColumnDimension()-removeColunas);
		
		/* Pega a submatriz de linhas 1 a m e colunas 1 a (n - removeColunas), sendo m e n
		a dimensao das linhas e colunas da matriz de dados fornecida como parametro. */
		resultado.getMatrix(1, dados.getRowDimension(), 1, dados.getRowDimension()-removeColunas);
		
		return resultado;
	}
	
	public static double[] calculaVariancia(Matrix dados){
		double[] medCol = new double[dados.getColumnDimension()];
		double[] auxCol = new double[dados.getColumnDimension()];
		double[] varCol = new double[dados.getColumnDimension()];
		
		medCol = calculaMedia(dados);
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			for (int linha = 0; linha < dados.getRowDimension(); linha++) {
				auxCol[coluna]= 
						auxCol[coluna] + ((dados.get(linha, coluna)-medCol[coluna]) * (dados.get(linha, coluna)-medCol[coluna]));
			}
			varCol[coluna] = auxCol[coluna]/((double)(dados.getRowDimension()-1));
		}
		
		return varCol;
	}
	
	public static double[] calculaMedia(Matrix dados){
		double[] medCol = new double[dados.getColumnDimension()];
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			medCol[coluna] = JamaUtils.colsum(dados, coluna)/dados.getRowDimension();
		}
		
		return medCol;
	}
	
	public static double[] calculaIntervalo(Matrix dados){
		double[] intervCol = new double[dados.getColumnDimension()];
		
		for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
			intervCol[coluna] = 
					JamaUtils.getMax(JamaUtils.getcol(dados, coluna)) - JamaUtils.getMin(JamaUtils.getcol(dados, coluna));
		}
		
		return intervCol;
	}
	
	
	public static void main(String[] args){
		//Situacao_Problema sit = Leitura_Arquivo.obtem_dados(args[0]);
		//Matrix demo = sit.get_entrada();
		Matrix demo2 = new Matrix(5, 5);
        
        /*for (int i = 0; i < demo2.getRowDimension(); i++) {
			for (int j = 0; j < demo2.getColumnDimension(); j++) {
				demo2.set(i, j, i+j);
			}
		}*/
		
		
		int aux = 0;
		for (int i = 0; i < demo2.getRowDimension(); i++) {
			for (int j = 0; j < demo2.getColumnDimension(); j++) {
				aux++;
				demo2.set(i, j, aux);
			}
		}
		
		
		/*
		double[] media = calculaMedia(demo2);
		for (int i = 0; i < media.length; i++) {
			System.out.println(media[i]);
		}
		*/
		
		double[] var = calculaVariancia(demo2);
		for (int i = 0; i < var.length; i++) {
			System.out.println(var[i]);
		}
		
		
		/*
		double[] desv = calculaVariancia(demo2);
		for (int i = 0; i < desv.length; i++) {
			System.out.println(Math.sqrt(desv[i]));
		}
		*/
		
		
		
		//demo2.print(1, 1);
        //Matrix teste1 = Pre_Processamento.sigmoidal(demo2);
        //teste1.print(1, 3);
		
		
		
	}
}