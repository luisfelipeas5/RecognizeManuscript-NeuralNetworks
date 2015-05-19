import Jama.Matrix;

abstract class Rede {
	int numero_neuronios_escondidos; // No caso da LVQ = 0 
	boolean treina_padrao_padrao; 
	boolean treina_batelada;
	
	Rede (int numero_neuronios_escondidos, boolean treina_padrao_padrao, boolean treina_batelada){
		this.numero_neuronios_escondidos=numero_neuronios_escondidos;
		this.treina_padrao_padrao=treina_padrao_padrao;
		this.treina_batelada=treina_batelada;
	}
	
	/*
	 *Esse metodo retorna o erro quadratico total da epoca, dado
	 *- um conjunto de entradas
	 *- um conjunto de saidas desejadas
	 */
	abstract double calcula_saida(Matrix entradas, Matrix saida_desejadas, Matrix pesos_a, Matrix pesos_b);
	/*
	 * Esse metodo atualiza as matrizes de pesos dado um erro
	 * manipulado pela classe calcula_saida
	 */
	abstract void atualiza_pesos(double erro, Matrix pesos_a, Matrix pesos_b);
}
