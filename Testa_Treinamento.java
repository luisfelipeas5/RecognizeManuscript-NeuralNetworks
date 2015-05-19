import Jama.Matrix;

public class Testa_Treinamento {
	public static void main(String[] args) {
		
		//Gera automaticamente entradas e saidas desejadas para efeito de teste
		Matrix entradas_sem_bias=Matrix.random(10, 4);//new Matrix(10,4);
		Matrix saidas_desejadas=Matrix.random(10,1);//new Matrix(10,1);
		
		//Adicionando o Bias
		Matrix entradas=new Matrix( entradas_sem_bias.getRowDimension(), entradas_sem_bias.getColumnDimension()+1 );
		for (int i = 0; i < entradas_sem_bias.getRowDimension(); i++) {
			Matrix entrada_sem_bias=entradas_sem_bias.getMatrix(i, i, 0, entradas_sem_bias.getColumnDimension()-1);
			entradas.setMatrix(i, i, 0, entradas.getColumnDimension()-2, entrada_sem_bias);
			//Acrescenta 1 como bias
			double bias=1;
			entradas.set(i, entradas.getColumnDimension()-1, bias);
		}

		Matrix entradas_treinamento;
		Matrix saidas_desejadas_treinamento;
		Matrix entradas_validacao;
		Matrix saidas_desejadas_validacao;
		Matrix entradas_teste;
		Matrix saidas_desejadas_teste;
		
		//60% treinamento 20% validacao 20% teste
 		double porcentagem_validacao=0.2;
 		double porcentagem_teste=0.2;
		
		//Definicao do conjunto de dados para treinamento
		int num_linhas_treinamento= (int) (entradas.getRowDimension() * (1-porcentagem_validacao-porcentagem_teste));
		int i_inicio_treinamento=0;
		int i_final_treinamento=i_inicio_treinamento+num_linhas_treinamento-1;
		entradas_treinamento=entradas.getMatrix(i_inicio_treinamento, i_final_treinamento,
			0, entradas.getColumnDimension()-1);
		saidas_desejadas_treinamento=saidas_desejadas.getMatrix(i_inicio_treinamento, i_final_treinamento,
			0, saidas_desejadas.getColumnDimension()-1);
		
		//Definicao do conjunto de dados para validacao
		int num_linhas_validacao= (int) (entradas.getRowDimension() * porcentagem_validacao);
		int i_inicio_validacao=i_final_treinamento+1;
		int i_final_validacao=i_inicio_validacao+num_linhas_validacao-1;
		entradas_validacao=entradas.getMatrix(i_inicio_validacao, i_final_validacao,
			0, entradas.getColumnDimension()-1) ;
		saidas_desejadas_validacao=saidas_desejadas.getMatrix(i_inicio_validacao, i_final_validacao,
			0, saidas_desejadas.getColumnDimension()-1) ;;
		
		//Definicao do conjunto de dados para teste
		int num_linhas_teste= (int) (entradas.getRowDimension() * porcentagem_teste);
		int i_inicio_teste=i_final_validacao+1;
		int i_final_teste=i_inicio_teste+num_linhas_teste-1;
		entradas_teste=entradas.getMatrix(i_inicio_teste, i_final_teste,
			0, entradas.getColumnDimension()-1) ;
		saidas_desejadas_teste=saidas_desejadas.getMatrix(i_inicio_teste, i_final_teste,
			0, saidas_desejadas.getColumnDimension()-1) ;;											
		
		//Definicao das configuracoes da rede
		int numero_neuronios_escondidos=2;
		
		//Definicao das condicoes do treinamento da rede
		boolean treina_padrao_padrao=true; //a rede ira ser treinada padrao a padrao (online)?
		boolean treina_batelada=!treina_padrao_padrao; //a rede ira ser treinada a batelada (batch)?
		int epocas_max=2; //numero de epocas maximas que a rede ira ser treinada
		
		Rede mlp=new MLP( numero_neuronios_escondidos, treina_padrao_padrao, treina_batelada);
		
		//Um objeto treinamento para cada tipo de rede
		Treinamento treinamento=new Treinamento(mlp);
		treinamento.treina(entradas_treinamento, saidas_desejadas_treinamento, entradas_validacao, saidas_desejadas_validacao, epocas_max);
		
		//Exibe matrizes geradas
		System.out.println("Matriz de entradas sem bias");
		entradas_sem_bias.print(entradas_sem_bias.getColumnDimension(), 3);
		System.out.println("Matriz de saidas desejadas");
		saidas_desejadas.print(saidas_desejadas.getColumnDimension(), 3);
		
		System.out.println("Matriz de entradas com bias");
		entradas.print(entradas.getColumnDimension(), 3);
		
		System.out.println("Matriz de entradas treinamento");
		entradas_treinamento.print(entradas_treinamento.getColumnDimension(), 3);
		System.out.println("Matriz de saidas desejadas treinamento");
		saidas_desejadas_treinamento.print(saidas_desejadas_treinamento.getColumnDimension(), 3);
		
		System.out.println("Matriz de entradas validacao");
		entradas_validacao.print(entradas_validacao.getColumnDimension(), 3);
		System.out.println("Matriz de saidas desejadas validacao");
		saidas_desejadas_validacao.print(saidas_desejadas_validacao.getColumnDimension(), 3);
		
		System.out.println("Matriz de entradas teste");
		entradas_teste.print(entradas_teste.getColumnDimension(), 3);
		System.out.println("Matriz de saidas desejadas teste");
		saidas_desejadas_teste.print(saidas_desejadas_teste.getColumnDimension(), 3);
		
		System.out.println("Configuracoes da rede MLP:");
		System.out.println("\tNumero de neuronios escondidos "+mlp.numero_neuronios_escondidos);
		System.out.println("Condicoes de treinamento da rede MLP:");
		System.out.println("\tTreinamento a batelada ("+mlp.treina_batelada+")");
		System.out.println("\tTreinamento a padrao a padrao ("+mlp.treina_padrao_padrao+")");
		
	}
}

