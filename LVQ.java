//nao estou usando o JAMA

import java.util.Random;

class LVQ extends Rede{
	
	public double distancia_euclidiana(double[] vetor1, double[] vetor2){
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
	
	//public static void main(String[]args){
	
		//LVQ so_distancia = new LVQ(2, 2, 1);
		
		/*double[] vetor_um = new double[so_distancia.dimensao];
		vetor_um[0] = 2;
		vetor_um[1] = 0;
		
		double[] vetor_dois = new double[so_distancia.dimensao];
		vetor_dois[0] = 10;
		vetor_dois[1] = 1;
		
		double distancia = so_distancia.distancia_euclidiana(vetor_um, vetor_dois);
		System.out.println(distancia);*/
		so_distancia.inicializa_pesos();
	//} Nao precisa de main
	
	
	Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b);
	 


}
