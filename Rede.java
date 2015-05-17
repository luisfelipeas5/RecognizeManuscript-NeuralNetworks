import Jama.Matrix;


abstract class Rede {
	int numero_neuronios_escondidos; // No caso da LVQ = 0
	int numero_neuronios_saida; // Varia de acordo com a codificacao da saida desejada
	double taxa_aprendizado;
	boolean treina_padrao_padrao; 
	Matrix entrada; 
	Matrix saida_desejada; 
	int numero_entradas;
	int numero_saidas;
	Matrix[] matrizes_pesos;
	Matrix[] saidas_rede;
	
	Rede (double taxa_aprendizado, int numero_neuronios_escondidos, boolean treina_padrao_padrao, Matrix saida, Matrix entrada){
		this.taxa_aprendizado=taxa_aprendizado;
		this.numero_neuronios_escondidos=numero_neuronios_escondidos;
		this.treina_padrao_padrao=treina_padrao_padrao;
		this.saida_desejada = saida;
		double[][] s = saida.getArrayCopy(); 
		double[][] e = entrada.getArrayCopy(); 
		this.numero_entradas = e[0].length; 
		this.numero_saidas = s[0].length; 
		matrizes_pesos = new Matrix[2];
	}
	
	/*
	 *Esse metodo calcula a saida dessa rede para uma instancia somente, 
	 *dado duas matrizes de peso  
	 */
	abstract Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b);
	/*
	 * Esse metodo atualiza as matrizes de pesos dado um erro
	 */
	abstract void atualiza_pesos(Double erro, Matrix pesos_a, Matrix pesos_b );
}
