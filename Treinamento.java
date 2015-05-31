import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import Jama.Matrix;

public class Treinamento {
	/*
	 * A classe treinamento eh responsavel pelo controle de epocas do treinamento da rede passada como
	 * parametro. Durante as epocas, guarda-se os erros referentes de treinamento e de validacao para
	 * comparacao futura. Alem disso eh responsavel por fazer o corte da LVQ, caso a rede do objeto seja
	 * uma.
	 */
	Rede rede;
	Matrix pesos_a;
	Matrix pesos_b;
	
	public Treinamento(Rede rede) {
		this.rede=rede;
	}

	/*
	 * Esse método treina a rede setada para o objeto
	 * passando como parametro o conjunto de dados para treinamento e para validacao,
	 * e utiliza, como condicao de parada do treinamento, um numero limite de epocas 
	 * e indica qual epoca seria ideal para parar o treinamento. Essa epoca ideal seria quando
	 * o erro de treinamento e erro da validacao sao iguais em determinada epoca. 
	 * O retorno do metodo eh uma matriz coluna com os erros quadraticos da rede 
	 * calculados pela propria rede em cada epoca que houve treinamento.
	 * O metodo de treinamento (batelada ou padrao a padrao) esta definido internamente na Rede,
	 * e foi decidido quando esta foi instanciada
	 */
	public Matrix treina(Matrix entradas_treinamento, Matrix saidas_desejadas_treinamento,
						Matrix entradas_validacao, Matrix saidas_desejadas_validacao, 
						int numero_limite_epocas, boolean pesos_aleatorios,
						double intervalo_pesos_aleatorios, double limiar_erro) {
		/*
		 * Define se as linhas dos pesos devem ser multiplcadas pelo numero de classes
		 * Multiplcacao pelo numero de neuronios por classe caso a rede seja uma LVQ.
		 * Caso seja uma MLP, o multiplcacao nao eh necessario
		 */
		boolean eh_mlp=false;
		int fator_multiplicacao=1;
		try {
			@SuppressWarnings("unused")
			MLP mlp=(MLP)rede; //Caso seja uma LVQ, uma excessao que o cast nao eh possivel eh lancada
			eh_mlp=true;
		}catch(ClassCastException cce){
			fator_multiplicacao = ((LVQ)rede).numero_de_classes;
		}
		
		Matrix pesos_a= new Matrix( rede.numero_neuronios*fator_multiplicacao, entradas_treinamento.getColumnDimension() );
		Matrix pesos_b= new Matrix( saidas_desejadas_treinamento.getColumnDimension(), rede.numero_neuronios+1 );
		
		if(pesos_aleatorios) { //caso seja setado para que os pesos sejam gerados randomicamente
			//Para a primeira epoca, os pesos devem ser gerados de forma randomica
			gera_pesos_aleatorios(pesos_a, intervalo_pesos_aleatorios);
			gera_pesos_aleatorios(pesos_b, intervalo_pesos_aleatorios);
		}
		
		//a rede recebe pesos definidos aqui. Se a Rede for LVQ, ela tratará os pesos b como null internamente
		rede.set_pesos(pesos_a, pesos_b);
		
		/*
		 * Armazenar erros total da rede de cada epoca:
		 * - erros do treinamento sao guardados na primeira coluna
		 * - erros de validacao sao guardados na segunda coluna 
		 * Na MLP, o erro vai ser quadratico total, e
		 * na LVQ, a taxa de erro ira ser a contagem de quantas instancias a rede errou na classificacao
		 * dividido pelo numero de intancias
		 */
		Matrix erros_epocas=new Matrix(numero_limite_epocas, 2);
		//Cada conjunto de dados vai ter o seu erro com o passar das épocas
		double erro_total_treinamento=1.0;
		double erro_total_validacao=-1.0;
		/*
		 * A rede para de ser treinada quando o erro total de treinamento por epoca
		 * eh menor que o erro definido como limiar; ou
		 * Se o numero de epocas ultrapassar o numero de epocas limite estabelecido externamente
		 * a funcao.
		 */
		int epoca_ideal=0; //Epoca na qual o erro de treinamento eh igual ou menor do que erro de validacao
		int epoca_atual=0;
		while (epoca_atual<numero_limite_epocas && erro_total_treinamento>limiar_erro) {
			/*
			 * Caso a rede seja uma MLP, a epoca ideal para parada eh guardada quando o
			 * erro de treinamento for menor ou igual ao erro de validacao
			 */
			if(eh_mlp) {
				if(erro_total_treinamento<=erro_total_validacao) {
					epoca_ideal=epoca_atual;
				}
			}
			//O conjunto de entradas sao embaralhadas antes de treinar a rede
			Holdout.embaralhar_conjuntos(entradas_treinamento, saidas_desejadas_treinamento);
			rede.set_problema(entradas_treinamento, saidas_desejadas_treinamento);
			
			//erro total para o conjunto de treinamento
			erro_total_treinamento=this.rede.get_erro();
			
			//O conjunto de entradas sao embaralhadas antes de calcular o erro de validacao
			Holdout.embaralhar_conjuntos(entradas_validacao, saidas_desejadas_validacao);
			rede.set_problema(entradas_validacao, saidas_desejadas_validacao);
			//erro total para o conjunto de validacao,nao eh feita nenhum tipo de atualizacao de pesos 
			rede.set_necessidade_atualizacao();
			erro_total_validacao=this.rede.get_erro();
			rede.set_necessidade_atualizacao();
			
			//Armazena erros de treinamento e validacao da epoca atual
			erros_epocas.set(epoca_atual, 0, erro_total_treinamento);
			erros_epocas.set(epoca_atual, 1, erro_total_validacao);
			
			//Status do treinamento:
			System.out.format("\nEpoca = %d: Erro do treinamento = %.5f ; Erro da validacao = %.5f\n", (epoca_atual+1), erro_total_treinamento, erro_total_validacao);
			//Passou-se uma epoca!
			epoca_atual+=1;
			//System.out.println("Parando na primeira época!");
			//System.exit(0);
		}
		
		/*
		 * Corte da LVQ - somente os neuronios mais ativados entre cada uma das classes sao mantidos como
		 * pesos para a rede.
		 */
		if(!eh_mlp) {
			LVQ lvq=(LVQ)rede;
			/*
			 * As classes existentes no conjunto de treinamento sao armazenadas e o indice referente
			 * a cada instacia de entrada que corresponde a essa classe
			 */
			Map<Double, List<Integer>> indices_instancias_classe_treinamento = Holdout.contar_numero_de_instancias(saidas_desejadas_treinamento);
			Double[] classes=indices_instancias_classe_treinamento.keySet().toArray( new Double[0]);
			Arrays.sort(classes);
			
			//O processo de corte eh feito para a todas as classes
			for (int i = 0; i < classes.length ; i++) {
				//Matrix que guarda somente instancias da classe i
				int numero_instancias_classe_i=indices_instancias_classe_treinamento.get(classes[i]).size(); //numero de instancias da classe[i]
				Matrix entradas_classe=new Matrix(numero_instancias_classe_i, entradas_treinamento.getColumnDimension());
				
				//Lista de indices da instancia de entrada do treinamento que corresponde aquela classe i
				List<Integer> indices_classe=indices_instancias_classe_treinamento.get(classes[i]);
				Iterator<Integer> iterator_indices_classe = indices_classe.iterator();
				/*
				 * Cria um novo conjunto de dados para fazer o corte da LVQ. Esse conjunto
				 * contera somente as entradas da classe iterada e as saidas da classe iterada
				 */
				int linha_vazia=0;
				while(iterator_indices_classe.hasNext()) {
					Integer indice = iterator_indices_classe.next();
					
					Matrix entrada=entradas_treinamento.getMatrix(indice, indice, 0, entradas_treinamento.getColumnDimension()-1);
					entradas_classe.setMatrix(linha_vazia, linha_vazia, 0, entradas_treinamento.getColumnDimension()-1, entrada);
					
					linha_vazia+=1;
				}
				int numero_ideal_de_neuronios=2;
				lvq.corte_de_neuronios(numero_ideal_de_neuronios, classes[i], entradas_classe);
			}
		}
		
		System.out.println("\tEpoca ideal = "+epoca_ideal);
		System.out.format("\tEpoca = %d: Erro do treinamento = %.5f ; Erro da validacao = %.5f \n", epoca_atual, erro_total_treinamento, erro_total_validacao);
		return erros_epocas;
	}
	
	/*
	 * Esse metodo, dado uma matriz pesos de dimensoes quaisquer, preenche pesos com valores aleatorios
	 */
	public static void gera_pesos_aleatorios(Matrix pesos, double intervalo_pesos_aleatorios) {
		Random random=new Random();
		for(int i=0; i< pesos.getRowDimension(); i++) {
			for(int j=0; j<pesos.getColumnDimension();j++) {
				double peso=random.nextDouble()-intervalo_pesos_aleatorios;	
				pesos.set(i, j, peso);
			}
		}
	}
}
