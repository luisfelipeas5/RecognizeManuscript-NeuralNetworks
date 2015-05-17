//nao estou usando o JAMA

import java.util.Random;

import Jama.Matrix;

class LVQ extends Rede{
	
	Matrix resultado;
	
	public double distancia_euclidiana(Matrix vetor1, Matrix vetor2){
		//nao precisa tirar a raiz quadrada
		double distancia =0;
		for(int j=0;j<dimensao;j++){
			distancia = distancia + Math.pow((vetor1[j] - vetor2[j]), 2);
		}
		return distancia;
	}
	
/*	public void inicializa_pesos(){	//aleatorio
		pesos = new double[numero_neuronios_entrada][dimensao];
		Random rd = new Random();
		for(int i = 0; i<numero_neuronios_entrada; i++){
			for(int j = 0; j<dimensao; j++){
				pesos[i][j] = rd.nextDouble();
			} 	
		}	
	}
	Matriz de peso ja sera passada ao criar o objeto
	*/
	
	public double diminui_taxa_de_aprendizado(double taxa_de_aprendizado_atual){
		double taxa_atualizada;
		taxa_atualizada = taxa_de_aprendizado_atual*0.9;
		return taxa_atualizada;
	}
	
	
	Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b){
		resultado = entrada.times(pesos_a);
		return resultado;
	}
	 


}
