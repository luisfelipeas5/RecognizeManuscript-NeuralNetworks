import Jama.Matrix;


abstract class Rede {
	int numero_neuronios_escondidos;
	int numero_neuronios_saida;
	boolean treina_padrao_padrao;
	
	Rede (int numero_neuronios_escondidos, boolean treina_padrao_padrao){
		this.numero_neuronios_escondidos=numero_neuronios_escondidos;
		this.treina_padrao_padrao=treina_padrao_padrao;
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
