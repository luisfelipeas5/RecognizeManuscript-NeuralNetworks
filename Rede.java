import Jama.Matrix;

/**Classe abstrata cujos metodos devem ser implementados pelas classes que a extendem (MLP e LVQ).*/
abstract class Rede {
	int numero_neuronios; // No caso da LVQ = 0
	boolean necessidade_atualizar_pesos=true;
	
	Rede (int numero_neuronios){
		this.numero_neuronios=numero_neuronios; 
	}
	
	/**Retorna o erro obtido durante a execucao de uma epoca completa. */
	abstract double get_erro(); 
	/**Retorna todas as saidas geradas pela rede durante uma epoca*/
	abstract Matrix get_saidas(); 
	/**Informa a rede qual o modo de treinamento desejado. A codificacao eh 1 para o treinamento 
	padrao a padrao e 2 para o em batelada */
	abstract void set_modo_treinamento (int modo_treinamento); 
	/**Passa para a rede as matrizes de pesos iniciais*/
	abstract void set_pesos (Matrix pesos_a, Matrix pesos_b); 
	/**Passa para a rede as matrizes que contem as entradas e as correspondentes saidas. */
	abstract void set_problema (Matrix entrada, Matrix saida_desejada); 		
	/** Inverte o valor da variavel booleana que indica se a rede precisa atualizar as matrizes de pesos. */
	void set_necessidade_atualizacao() {
		necessidade_atualizar_pesos=!necessidade_atualizar_pesos;
	}
}
