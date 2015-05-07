
public class TestaTreinamento {
	public static void main(String[] args) {
		Treinamento treinamento=new Treinamento();
		double[][] entrada= 	{{1, 1, 1},
							{ 1, 1, 1 }};
		double[][] saidaDesejada= {{ 1 },
								{ 1 }};
		int numeroNeuroniosEscondidos=2;
		int epocas=2;
		treinamento.treinaMLP(entrada, saidaDesejada, numeroNeuroniosEscondidos, epocas);
	}
}
