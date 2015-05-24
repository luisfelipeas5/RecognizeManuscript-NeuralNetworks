import java.util.Random;

import Jama.Matrix;

public class Treinamento {
	
	Rede rede;
	Matrix pesos_a;
	Matrix pesos_b;
	
	public Treinamento(Rede rede) {
		this.rede=rede;
	}

	/*
	 * Esse método treina a rede setada para a instância da classe
	 * passando como o conjunto de dados para treinamento a matriz passada como parâmetro,
	 * e utilizando como condicao de parada um numero limite de epocas ou quando os erros de validacao
	 * e treinamento sao iguais em determinada epoca, retornando uma matriz coluna com
	 * os erros quadraticos da rede calculados pela propria rede em cada epoca que houve treinamento.
	 * O metodo de treinamento esta definido internamente na Rede, e foi decidido quando esta foi
	 * instanciada
	 */
	public Matrix treina(Matrix entradas_treinamento, Matrix saidas_desejadas_treinamento,
						Matrix entradas_validacao, Matrix saidas_desejadas_validacao, 
						int numero_limite_epocas, boolean pesos_aleatorios) {
		System.out.println("\n\n#--------------começo da Fase de Treinamento---------------#");
		System.out.println("\t\tNumero de limite de epocas="+numero_limite_epocas);
		
		/*
		 * para saber se as linhas dos pesos devem ser multiplcadas pelo numero de classes
		 * se a rede do treinamento eh uma LVQ. Caso seja uma MLP, o multiplcacao nao eh necessario
		 */
		int fator_multiplicacao=1;
		try {
			MLP mlp=(MLP)rede; //Caso seja uma LVQ, uma excessao que o cast nao eh possivel eh lancada
			mlp.set_necessidade_atualizacao();
		}catch(ClassCastException cce){
			fator_multiplicacao = ((LVQ)rede).numero_de_classes;
		}
		
		Matrix pesos_a= new Matrix( rede.numero_neuronios*fator_multiplicacao, entradas_treinamento.getColumnDimension() );
		Matrix pesos_b= new Matrix( saidas_desejadas_treinamento.getColumnDimension(), rede.numero_neuronios+1 );
		
		if(pesos_aleatorios) {
			//Para a primeira epoca, os pesos devem ser gerados de forma randomica
			this.gera_pesos_aleatorios(pesos_a);
			this.gera_pesos_aleatorios(pesos_b);
		}
		
		//a rede recebe pesos definidos aqui. Se a Rede for LVQ, ela tratará os pesos b como null internamente
		rede.set_pesos(pesos_a, pesos_b);
		
		/*
		 * Armazenar erros total da rede de cada epoca:
		 * - erros do treinamento sao guardados na primeira coluna
		 * - erros de validacao sao guardados na segunda coluna 
		 * Na MLP, o erro vai ser quadratico total, e
		 * na LVQ, o erro ira ser a contagem de quantas instancias a rede errou na classificacao
		 */
		Matrix erros_epocas=new Matrix(numero_limite_epocas, 2);
		//Cada conjunto de dados vai ter o seu erro com o passar das épocas
		double erro_total_treinamento=1.0;
		double erro_total_validacao=-1.0;
		/*
		 * A rede para de ser treinada quando o erro total por epoca 
		 * do conjunto de treinamento eh igual ou menor ao erro do conjunto de validacao; ou
		 * Se o numero de epocas ultrapassar o numero de epocas limite estabelecido externamente
		 * a funcao.
		 */
		int epoca_atual=0;
		//erro_total_treinamento>erro_total_validacao &&
		while ( epoca_atual<numero_limite_epocas) {
			System.out.println("epoca="+epoca_atual+"-> e(treinamento)="+erro_total_treinamento+" e(validacao)="+erro_total_validacao);
			
			rede.set_problema(entradas_treinamento, saidas_desejadas_treinamento);
			
			//((MLP) rede).entrada_completa.print(((MLP) rede).entrada_completa.getColumnDimension(), 3);
			//((MLP) rede).saida_desejada_completa.print(((MLP) rede).saida_desejada_completa.getColumnDimension(), 3);
			//((MLP) rede).pesos_a.print(((MLP) rede).pesos_a.getColumnDimension(), 3);
			//((MLP) rede).pesos_b.print(((MLP) rede).pesos_b.getColumnDimension(), 3);
			
			//erro total para o conjunto de treinamento
			erro_total_treinamento=this.rede.get_erro();
			
			rede.set_problema(entradas_validacao, saidas_desejadas_validacao);
			//erro total para o conjunto de validacao,nao eh feita nenhum tipo de atualizacao de pesos 
			rede.set_necessidade_atualizacao();
			erro_total_validacao=this.rede.get_erro();
			rede.set_necessidade_atualizacao();
			
			//Armazena erros de treinamento e validacao da epoca atual
			erros_epocas.set(epoca_atual, 0, erro_total_treinamento);
			erros_epocas.set(epoca_atual, 1, erro_total_validacao);
			
			//Passou-se uma epoca!
			epoca_atual+=1;
			
			/* Printar os pesos a cada iteracao */
			int casas_decimais=3;
			//System.out.println("Pesos A");
			//pesos_a.print(pesos_a.getColumnDimension(), casas_decimais);
			//System.out.println("Pesos B");
			//pesos_b.print(pesos_b.getColumnDimension(), casas_decimais);
		}
		
		System.out.println("\tTreinameno parou na: epoca="+epoca_atual+"-> e(treinamento)="+erro_total_treinamento+" e(validacao)="+erro_total_validacao);
		System.out.println("#---------------------termino da Fase de Treinamento------------------------------#");
		return erros_epocas;
	}
	
	/*
	 * Esse metodo, dado uma matriz pesos de dimensoes quaisquer, preenche pesos com valores aleatorios
	 * entre -1.0 e 1.0 
	 */
	public void gera_pesos_aleatorios(Matrix pesos) {
		//TODO escolher o intervalo de valores dos pesos
		Random random=new Random();
		for(int i=0; i< pesos.getRowDimension(); i++) {
			for(int j=0; j<pesos.getColumnDimension();j++) {
				if(i%2==0) {
					pesos.set(i, j, random.nextDouble());
				}else {
					pesos.set(i, j, -random.nextDouble());
				}
				
			}
		}
	}
}
