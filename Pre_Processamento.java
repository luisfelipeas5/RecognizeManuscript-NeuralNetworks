import Jama.Matrix;
import edu.umbc.cs.maple.utils.*;
// http://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/edu/umbc/cs/maple/utils/JamaUtils.html
import optimization.*;
// http://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/edu/umbc/cs/maple/utils/JamaUtils.html

public class Pre_Processamento{
	
	
	public static Matrix minmax(Matrix dados, int colClasse){
		double[] minCol;
		double[] maxCol;
		double[] intervCol;
		int coluna = 0;
		int linha = 0;
		Matrix resultado;
		Matrix col;
		
		//resultado.copy(dados);
		/*
		if(){
			
		}else{
			
		}
		*/
		
		while(coluna < dados.getColumnDimension()){
			
			minCol[coluna] = JamaUtils.getMin(JamaUtils.getcol(dados, coluna));
			maxCol[coluna] = JamaUtils.getMin(JamaUtils.getcol(dados, coluna));
			
			intervCol[coluna] = maxCol[coluna] - minCol[coluna];
			coluna++;
		}
		coluna = 1;
		double[][] dados_double = dados.getArrayCopy();
		double[][] resultado_double;
		
		while(coluna < dados.getColumnDimension()){
			while(linha < dados.getRowDimension() ){
				if(intervCol[coluna]!=0){
					resultado.set(linha, coluna, ((dados_double[linha][coluna] - minCol[coluna])/intervCol[coluna])); 
				}
				else{
					resultado.set(linha, coluna, 0);
				}
				linha++;
			}
			linha = 1;
			coluna++;
		}
		return resultado;
	}
	
	public Matrix max(Matrix dados, int colClasse){
		
		double[] maxCol;
		double[] medCol;
		double[] desCol;
		int coluna = 0;
		int linha = 0;
		Matrix resultado;
		Matrix col;
		//resultado.copy(dados);
		/*
		if(){
			
		}else{
			
		}
		*/
		
		while(coluna < dados.getColumnDimension()){
			maxCol[coluna] = JamaUtils.getMin(JamaUtils.getcol(dados, coluna));
			coluna++;
		}
		coluna = 1;
		double[][] dados_double = dados.getArrayCopy();
		
		while(coluna < dados.getColumnDimension()){
			while(linha < dados.getRowDimension() ){
				if(intervCol[coluna]!=0){
					resultado.set(linha, coluna, ((dados_double[linha][coluna]/maxCol[coluna]))); 
				}
				else{
					resultado.set(linha, coluna, 0);
				}
				linha++;
			}
			linha = 1;
			coluna++;
		}
		return resultado;
	}
	
	public Matrix zscore(Matrix dados, int colClasse){
		
		double[] minCol;
		double[] maxCol;
		double[] intervCol;
		double[] medCol;
		double[] desCol;
		int coluna = 0;
		int linha = 0;
		Matrix resultado;
		Matrix col;
		
		//resultado.clone(dados);
		/*
		if(){
			
		}else{
			
		}
		*/
		while(coluna < dados.getColumnDimension()){
			medCol[coluna] = JamaUtils.colsum(dados, coluna)/dados.getRowDimension();
			coluna++;
		}
		coluna = 1;
		while(coluna < dados.getColumnDimension()){
			medCol[coluna] = JamaUtils.colsum(dados, coluna)/dados.getRowDimension();
			desCol[coluna] = 
			coluna++;
		}
		coluna = 1;
		double[][] dados_double = dados.getArrayCopy();
		
		while(coluna < dados.getColumnDimension()){
			while(linha < dados.getRowDimension() ){
				if(intervCol[coluna]!=0){
					resultado.set(linha, coluna, ((dados_double[linha][coluna]/maxCol[coluna]))); 
				}
				else{
					resultado.set(linha, coluna, 0);
				}
				linha++;
			}
			linha = 1;
			coluna++;
		}
		return resultado;
	}
	
	
}
