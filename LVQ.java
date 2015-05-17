//nao estou usando o JAMA

import Jama.Matrix;
// http://math.nist.gov/javanumerics/jama/doc/Jama/Matrix.html

class LVQ extends Rede{
	
	Matrix resultado;
	
	public Matrix distancia_euclidiana(Matrix saida_desejada){
		
		Matrix distancia = resultado.minus(saida_desejada);

		distancia = distancia.arrayTimes(distancia);
		
		return distancia;
	}
	
	
	public double diminui_taxa_de_aprendizado(double taxa_de_aprendizado_atual){
		double taxa_atualizada;
		taxa_atualizada = taxa_de_aprendizado_atual*0.9;
		return taxa_atualizada;
	}
	
	
	Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b){
		resultado = entrada.times(pesos_a);
		return resultado;
	}
	
	abstract void atualiza_pesos(Double erro, Matrix pesos_a, Matrix pesos_b ){
		
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
	
	// http://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/edu/umbc/cs/maple/utils/JamaUtils.html

}
