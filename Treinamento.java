import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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
		
		if(pesos_aleatorios) {
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
		int epoca_ideal=0; //Epoca na qual o erro de treinamento eh igual ou menor do que erro de validacao
		int epoca_atual=0;
		while (epoca_atual<numero_limite_epocas) {
			//Condicao de parada exclusica para MLP
			if(eh_mlp) {
				if(erro_total_treinamento<=erro_total_validacao) {
					epoca_ideal=epoca_atual;
				}
			}
			Holdout.embaralhar_conjuntos(entradas_treinamento, saidas_desejadas_treinamento);
			rede.set_problema(entradas_treinamento, saidas_desejadas_treinamento);
			
			//erro total para o conjunto de treinamento
			erro_total_treinamento=this.rede.get_erro();
			
			Holdout.embaralhar_conjuntos(entradas_validacao, saidas_desejadas_validacao);
			rede.set_problema(entradas_validacao, saidas_desejadas_validacao);
			//rede.set_problema(entradas_treinamento, saidas_desejadas_treinamento);
			//erro total para o conjunto de validacao,nao eh feita nenhum tipo de atualizacao de pesos 
			rede.set_necessidade_atualizacao();
			erro_total_validacao=this.rede.get_erro();
			rede.set_necessidade_atualizacao();
			
			//Armazena erros de treinamento e validacao da epoca atual
			erros_epocas.set(epoca_atual, 0, erro_total_treinamento);
			erros_epocas.set(epoca_atual, 1, erro_total_validacao);
			
			//Status do treinamento:
			System.out.format("\nEpoca = %d: Erro do treinamento = %.5f ; Erro da validacao = %.5f", (epoca_atual+1), erro_total_treinamento, erro_total_validacao);
			//Passou-se uma epoca!
			epoca_atual+=1;
		}
		
		//Corte da LVQ
		if(!eh_mlp) {
LVQ lvq=(LVQ)rede;
			
			//Classes existentes no conjunto de treinamento
			Map<Double, List<Integer>> indices_instancias_classe_treinamento = Holdout.contar_numero_de_instancias(saidas_desejadas_treinamento);
			Double[] classes=indices_instancias_classe_treinamento.keySet().toArray( new Double[0]);
			Arrays.sort(classes);
			
			for (int i = 0; i < classes.length ; i++) {
				//Matrix que guarda somente instancias da classe i
				int numero_instancias_classe_i=indices_instancias_classe_treinamento.get(classes[i]).size(); //numero de instancias da classe[i]
				Matrix entradas_classe=new Matrix(numero_instancias_classe_i, entradas_treinamento.getColumnDimension());
				Matrix saidas_desejadas_classe=new Matrix(numero_instancias_classe_i, saidas_desejadas_treinamento.getColumnDimension());
				//Lista do indice da linha em que a instancia de uma determinada classe esta no conjunto de treinamento
				List<Integer> indices_classe=indices_instancias_classe_treinamento.get(classes[i]);
				Iterator<Integer> iterator_indices_classe = indices_classe.iterator();
				int linha_vazia=0;
				while(iterator_indices_classe.hasNext()) {
					Integer indice = iterator_indices_classe.next();
					Matrix entrada=entradas_treinamento.getMatrix(indice, indice, 0, entradas_treinamento.getColumnDimension()-1);
					entradas_classe.setMatrix(linha_vazia, linha_vazia, 0, entradas_treinamento.getColumnDimension()-1, entrada);
					linha_vazia+=1;
				}
				lvq.corte_de_neuronios(2, classes[i], entradas_classe, saidas_desejadas_classe);
			}
		}
		System.out.println("\tEpoca ideal = "+epoca_ideal);
		System.out.format("\tEpoca = %d: Erro do treinamento = %.2f ; Erro da validacao = %.2f", epoca_atual, erro_total_treinamento, erro_total_validacao);
		return erros_epocas;
	}
	
	/*
	 * Esse metodo, dado uma matriz pesos de dimensoes quaisquer, preenche pesos com valores aleatorios
	 * entre -1.0 e 1.0 
	 */
	public static void gera_pesos_aleatorios(Matrix pesos, double intervalo_pesos_aleatorios) {
		//TODO escolher o intervalo de valores dos pesos
		Random random=new Random();
		for(int i=0; i< pesos.getRowDimension(); i++) {
			for(int j=0; j<pesos.getColumnDimension();j++) {
				double peso=random.nextDouble()-intervalo_pesos_aleatorios;	
				pesos.set(i, j, peso);
			}
		}
	}
}
