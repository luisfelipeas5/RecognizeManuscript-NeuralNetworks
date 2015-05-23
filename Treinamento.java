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
			Matrix entradas_validacao, Matrix saidas_desejadas_validacao, int numero_limite_epocas) {
		//Para a primeira epoca, os pesos devem ser gerados randomicamente
		Matrix pesos_a= new Matrix( rede.numero_neuronios_escondidos, entradas_treinamento.getColumnDimension() );
		Matrix pesos_b= new Matrix( saidas_desejadas_treinamento.getColumnDimension(), rede.numero_neuronios_escondidos );
		this.gera_pesos_aleatorios(pesos_a);
		this.gera_pesos_aleatorios(pesos_b);
		
		/*
		 * Armazenar erros quadraticos totais da rede de cada epoca:
		 * - erros do treinamento sao guardados na primeira coluna
		 * - erros de validacao sao guardados na segunda coluna 
		 */
		Matrix erros_epocas=new Matrix(numero_limite_epocas, 2);
		//Cada conjunto de dados vai ter o seu erro com o passar das épocas
		double erro_total_treinamento=1.0;
		double erro_total_validacao=-1.0;
		
		/*
		 * A rede para de ser treinada quando o erro total por epoca 
		 * do conjunto de treinamento eh igual ao erro do conjunto de validacao; ou
		 * Se o numero de epocas ultrapassar o numero de epocas estabelecido externamente
		 * a funcao.
		 */
		int epoca_atual=0;
		while (erro_total_treinamento>erro_total_validacao && epoca_atual<numero_limite_epocas) {
			//erro total para o conjunto de treinamento
			erro_total_treinamento= this.rede.calcula_saida(entradas_treinamento, saidas_desejadas_treinamento, pesos_a, pesos_b);
			//erro_total_treinamento= this.rede.get_erro_epoca(entradas_treinamento, saidas_desejadas_treinamento, pesos_a, pesos_b);
			
			/*erro total para o conjunto de validacao
				*nao eh feita nenhum tipo de atualizacao de pesos, por isso
				*a rede nao pode atualizar padrao a padrao, nem mesmo em batch 
			*/		
			boolean treina_batelada_old=this.rede.treina_batelada;
			boolean treina_padrao_padrao_old=this.rede.treina_padrao_padrao; //armazena o antigo tipo de treinamento da rede antes de mudar 
			this.rede.treina_padrao_padrao=false;
			this.rede.treina_batelada=false;
			erro_total_validacao= this.rede.calcula_saida(entradas_validacao, saidas_desejadas_validacao, pesos_a, pesos_b);
			this.rede.treina_batelada=treina_batelada_old;
			this.rede.treina_padrao_padrao=treina_padrao_padrao_old;
			
			//Armazena erros de treinamento e validacao da epoca atual
			erros_epocas.set(epoca_atual, 0, erro_total_treinamento);
			erros_epocas.set(epoca_atual, 1, erro_total_validacao);
			//Passou-se uma epoca!
			epoca_atual+=1;
			
			/* Printar os pesos a cada iteracao */
			System.out.println("Epoca "+epoca_atual);
			int casas_decimais=3;
			System.out.println("Pesos A");
			pesos_a.print(pesos_a.getColumnDimension(), casas_decimais);
			System.out.println("Pesos B");
			pesos_b.print(pesos_b.getColumnDimension(), casas_decimais);
		}
		
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
