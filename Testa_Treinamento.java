import Jama.Matrix;

public class Testa_Treinamento {
	public static void main(String[] args) {
		Treinamento treinamento=new Treinamento();
		double[][] array_entrada= 	{{1, 1, 1},
									{ 1, 1, 1 }};
		Matrix entrada=new Matrix(array_entrada);
		
		double[][] array_saida_desejada= {{ 1 },
										 { 1 }};
		Matrix saida_desejada=new Matrix(array_saida_desejada);
		
		int numero_neuronios_escondidos=2;
		int epocas=2;
		treinamento.treina_mlp(entrada, saida_desejada, numero_neuronios_escondidos, epocas);
	}
}

