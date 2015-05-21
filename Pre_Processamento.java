import Jama.Matrix;
import edu.umbc.cs.maple.utils.*;
import optimization.*;
// http://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/edu/umbc/cs/maple/utils/JamaUtils.html
// http://www.seas.upenn.edu/~eeaton/software.html
public class Pre_Processamento{
	
	public static Matrix max(Matrix dados){
		double[] maxCol;
		Matrix resultado;
		Matrix col;

		for(int coluna = 1; coluna <= dados.getColumnDimension(); coluna++){
			maxCol[coluna] = JamaUtils.getMax(JamaUtils.getcol(dados, coluna));
		}
		
		for (int coluna = 1; coluna <= dados.getColumnDimension(); coluna++) {
			for (int linha = 1; linha <= dados.getRowDimension(); linha++) {
				if(maxCol[coluna]!=0){
					resultado.set(linha, coluna, dados.get(linha, coluna)/maxCol[coluna]);
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		return resultado;
	}
	
	public static Matrix minmax(Matrix dados){
		double[] intervCol;
		Matrix resultado;
		
		intervCol = calculaIntervalo(dados);
		
		for (int coluna = 1; coluna <= dados.getColumnDimension(); coluna++) {
			for (int linha = 1; linha <= dados.getRowDimension(); linha++) {
				if(intervCol[coluna]!=0){
					resultado.set(linha, coluna, ((dados.get(linha, coluna) - minCol[coluna])/intervCol[coluna])); 
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		
		return resultado;
	}
	
	public static Matrix sigmoidal(Matrix dados){
		double[] intervCol;
		Matrix resultado;
		
		intervCol = calculaIntervalo(dados);
		
		for (int coluna = 1; coluna <= dados.getColumnDimension(); coluna++) {
			for (int linha = 1; linha <= dados.getRowDimension(); linha++) {
				if(intervCol[coluna]!=0){
					resultado.set(linha, coluna, -1/(1+Math.exp(-((dados.get(linha, coluna) - minCol[coluna])/intervCol[coluna])))); 
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		
		return resultado;
	}
	
	public static Matrix zscore(Matrix dados){
		
		double[] medCol;
		double[] varCol;
		Matrix resultado;
		Matrix col;
		
		medCol = calculaMedia(dados);
		varCol = calculaVariancia(dados);
		
		for(int coluna = 1; coluna <= dados.getColumnDimension(); coluna++){
			for(int linha = 1; linha <= dados.getRowDimension(); linha++){
				if(varCol[coluna]!=0){
					resultado.set(linha, coluna, dados.get(linha, coluna)-medCol[coluna]/Math.sqrt(varCol[coluna]));
				}
				else{
					resultado.set(linha, coluna, 0);
				}
			}
		}
		
		return resultado;
	}
	
	public static Matrix removeZeros(Matrix dados, int porcentagem){
		// Corte eh calculado a partir da porcentagem estipulada durante a chamada do metodo
		// Ex: Se deseja-se eliminar todas as colunas compostas 100% de zeros, basta passar
		// int porcentagem = 100. Com isso, corte = numero de linhas das colunas.
		int corte = (porcentagem/100)*dados.getRowDimension();
		int contaZeros = 0;
		Matrix resultado;
		
		for (int coluna = 1; coluna <= dados.getColumnDimension(); coluna++) {
			for (int linha = 1; linha < dados.getRowDimension(); linha++) {
				if(dados.get(linha, coluna) == 0){
					contaZeros++;
				}
			}
			// Se a quantidade de zeros for menor que o corte, adiciona essa coluna
			if(contaZeros < corte){
				for (int linha = 1; linha < dados.getRowDimension(); linha++) {
						resultado.set(linha, coluna, dados.get(linha, coluna));
				}
			}
			contaZeros = 0;
		}
		
		return resultado;
	}
	
	public static Matrix removeDesvioBaixo(Matrix dados, int limiar){
		// Corte eh calculado a partir da porcentagem estipulada durante a chamada do metodo
		// Ex: Se deseja-se eliminar todas as colunas compostas 100% de zeros, basta passar
		// int porcentagem = 100. Com isso, corte = numero de linhas das colunas.
		double[] varCol;
		Matrix resultado;
		
		varCol = calculaVariancia(dados);
		
		for (int coluna = 1; coluna <= dados.getColumnDimension(); coluna++) {
			for (int linha = 1; linha < dados.getRowDimension(); linha++) {
				if(Math.sqrt(varCol[coluna]) > limiar){
					resultado.set(linha, coluna, dados.get(linha, coluna));
				}
			}
		}
		
		return resultado;
	}
	
	public static double[] calculaVariancia(Matrix dados){
		double[] medCol;
		double[] auxCol;
		double[] varCol;
		
		medCol = calculaMedia(dados);
		
		for (int coluna = 1; coluna <= dados.getColumnDimension(); coluna++) {
			for (int linha = 1; linha <= dados.getRowDimension(); linha++) {
				auxCol[coluna]= 
						auxCol[coluna] + (dados.get(linha, coluna)-medCol[coluna]) * (dados.get(linha, coluna)-medCol[coluna]);
			}
			 
			varCol[coluna] = auxCol[coluna]/medCol[coluna];
		}
		
		return varCol;
	}
	
	public static double[] calculaMedia(Matrix dados){
		double[] medCol;
		
		for (int coluna = 1; coluna < dados.getColumnDimension(); coluna++) {
			medCol[coluna] = JamaUtils.colsum(dados, coluna)/dados.getRowDimension();
		}
		
		return medCol;
	}
	
	public static double[] calculaIntervalo(Matrix dados){
		double[] minCol;
		double[] maxCol;
		double[] intervCol;
		
		for (int coluna = 1; coluna < dados.getColumnDimension(); coluna++) {
			minCol[coluna] = JamaUtils.getMin(JamaUtils.getcol(dados, coluna));
			maxCol[coluna] = JamaUtils.getMax(JamaUtils.getcol(dados, coluna));
			intervCol[coluna] = maxCol[coluna] - minCol[coluna];
		}
		
		return intervCol;
	}
}