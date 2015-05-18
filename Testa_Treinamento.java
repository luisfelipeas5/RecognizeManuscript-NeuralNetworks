import Jama.Matrix;

public class Testa_Treinamento {
	public static void main(String[] args) {
		double[][] array_entrada= 	{{1, 1, 1},
									{ 1, 1, 1 }};
		Matrix entradas=new Matrix(array_entrada);
		
		double[][] array_saida_desejada= {{ 1 },
										 { 1 }};
		Matrix saidas_desejadas=new Matrix(array_saida_desejada);
		
		int numero_neuronios_escondidos=2;
		//int epocas=2; 
		/*
		o numero de epocas na verdade eh definido no ponto de interseccao quando:
		validacao.erro==treinamento.erro
		*/
		//treinando padrao a padrao:
		boolean treina_padrao_padrao=true;
		Rede mlp=new MLP( numero_neuronios_escondidos, treina_padrao_padrao);
		
		//70% treinamento 30% validacao
		double validacao=0.3;
		int num_linhas_validacao= (int) (entradas.getRowDimension() * validacao);
		Matrix entradas_validacao=entradas.getMatrix(entradas.getRowDimension()-num_linhas_validacao, num_linhas_validacao-1,0, entradas.getColumnDimension()) ;
		Matrix saidas_desejadas_validacao=saidas_desejadas.getMatrix(entradas.getRowDimension()-num_linhas_validacao, num_linhas_validacao-1, 0, entradas.getColumnDimension()) ;;
		
		//Um objeto treinamento para cada tipo de rede
		Treinamento treinamento=new Treinamento(mlp);
		treinamento.treina(entradas, saidas_desejadas, entradas_validacao, saidas_desejadas_validacao);
	}
}

