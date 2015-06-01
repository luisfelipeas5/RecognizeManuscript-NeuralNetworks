import Jama.Matrix; 

/** Serve apenas para compactar as matrizes de entrada e saida esperada, obtidas a partir da leitura do arquivo de texto.*/
public class Situacao_Problema {
	Matrix entrada; 
	Matrix saida; 
	int[] dim_entrada; 
	int[] dim_saida; 
	
	public Situacao_Problema (double[][] dados_entrada, double[][] dados_saida) {
		entrada = new Matrix (dados_entrada); 
		saida = new Matrix (dados_saida);
		dim_entrada = new int[2]; 
		dim_saida = new int[2]; 
		dim_entrada[0] = dados_entrada.length; 
		dim_entrada[1] = dados_entrada[0].length; 
		dim_saida[0] = dados_saida.length; 
		dim_saida[1] = dados_saida[0].length; 
	}
	
	/**Retorna uma matriz que contem todas as instancias de entrada. */
	public Matrix get_entrada() {
		return entrada; 
	}
	
	/**Retorna uma matriz que contem todas as instancias de saida esperada.*/
	public Matrix get_saida() {
		return saida; 
	}
	
	/**Retorna o numero de entradas. */
	public int numero_entradas() {
		return dim_entrada[1]; 
	}
	
	/**Retorna o numero de saidas. */
	public int numero_saidas() {
		return dim_saida[1];  
	}
	
	/**Retorna o numero de instancias. */
	public int numero_instancias() {
		return dim_entrada[0]; 
	}
}
