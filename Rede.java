import Jama.Matrix;


abstract class Rede {
	int numero_neuronios_escondidos; // No caso da LVQ = 0 
	boolean treina_padrao_padrao; 
	
	//Matrix entrada; 
	//Matrix saida_desejada; 
	//Matrix[] matrizes_pesos;
	//Matrix[] saidas_rede;
	//int numero_neuronios_saida; = saida.getColumnDimension() // Varia de acordo com a codificacao da saida desejada
	//int numero_entradas; = entrada.getColumnDimension()
	//int numero_saidas; = saida.getColumnDimension()
	double taxa_aprendizado;
	
	Rede (int numero_neuronios_escondidos, boolean treina_padrao_padrao){
		this.numero_neuronios_escondidos=numero_neuronios_escondidos;
		this.treina_padrao_padrao=treina_padrao_padrao;
		/*
		double[][] s = saida.getArrayCopy(); 
		double[][] e = entrada.getArrayCopy(); 
		this.numero_entradas = e[0].length; 
		this.numero_saidas = s[0].length;
		*/
	}
	
	/*
	 *Esse metodo calcula a saida dessa rede para uma instancia somente, 
	 *dado duas matrizes de peso  
	 */
	abstract Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b);
	/*
	 * Esse metodo atualiza as matrizes de pesos dado um erro
	 */
	abstract void atualiza_pesos(double erro, Matrix pesos_a, Matrix pesos_b, double taxa_aprendizado );
}
