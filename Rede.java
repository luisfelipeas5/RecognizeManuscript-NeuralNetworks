import Jama.Matrix;

abstract class Rede {
	int numero_neuronios; // No caso da LVQ = 0
	
	Rede (int numero_neuronios){
		this.numero_neuronios=numero_neuronios; 
	}
	
	/*
	 *Esse metodo retorna o erro quadratico total da epoca, dado
	 *- um conjunto de entradas
	 *- um conjunto de saidas desejadas
	 */
	//abstract void calcula_saida(Matrix entrada, Matrix saida_desejadas, );
	/*
	 * Esse metodo atualiza as matrizes de pesos dado um erro
	 * manipulado pela classe calcula_saida
	 */
	//abstract void atualiza_pesos(Matrix entrada, Matrix saida, Matrix pesos_a, Matrix pesos_b);
	abstract double get_erro(int modo_treinamento); 
	abstract Matrix get_saidas(); 
	abstract void set_pesos (Matrix pesos_a, Matrix pesos_b); 
	abstract void set_problema (Matrix entrada, Matrix saida_desejada); 		
}
