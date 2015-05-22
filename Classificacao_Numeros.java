import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Jama.Matrix;


public class Classificacao_Numeros {
	static Matrix entradas;
	static Matrix saidas_desejadas;
	
	static Matrix entradas_treinamento;
	static Matrix saidas_desejadas_treinamento;
	static Matrix entradas_validacao;
	static Matrix saidas_desejadas_validacao;
	static Matrix entradas_teste;
	static Matrix saidas_desejadas_teste;
	
	static Map<Double,List<Integer>> indices_instancias_classe; //Map que armazena em quais indices estao cada uma dos valores de saida desejadas
	
	public static void main(String[] args) {
		//Estabelece o nome do arquivo que contem o conjunto de dados
		String nome_arquivo_conjunto_dados;
		nome_arquivo_conjunto_dados="conjunto_dados.txt";
		//nome_arquivo_conjunto_dados="optdigits.total.txt";
		
		/*
		 * Le o arquivo do conjunto de dados e separa em:
		 * 	- Atributos (Entradas); e
		 *  - Atributos Classe (Saidas Desejadas).
		 * Colocando-os em matrizes
		 */
		Situacao_Problema situacao_problema_conjunto_dados = Leitura_Arquivo.obtem_dados(nome_arquivo_conjunto_dados);
		Matrix entradas_sem_bias=situacao_problema_conjunto_dados.get_entrada();
		saidas_desejadas=situacao_problema_conjunto_dados.get_saida();
		
		/*
		 * Adiciona à matriz de entrada as entradas referentes ao bias!
		 */
		entradas=adiciona_bias(entradas_sem_bias);//Matriz das instâncias de entrada com bias
		
		separa_conjuntos();
		
		int numero_epocas=10;
		
		int numero_neuronios_escondidos=2;
		Rede mlp=null;
		//mlp=new MLP(numero_neuronios_escondidos, taxa_aprendizado_inicial);
		matriz_confusao(mlp);
		
		int numero_neuronios_classes=3;
		Rede lvq=null;
		//lvq=new LVQ(numero_neuronios_classes);
	}
	
	public static void separa_conjuntos() {
		/*
		 * Separacao do conjunto de entradas em conjunto de :
		 *  - treinamento ( 1-(porcentagem_validacao+porcentagem_teste)% do conjunto de entradas)
		 *  - validacao   ( porcentagem_validacao% do conjunto de entradas)
		 *  - teste       ( porcentagem_teste% do conjunto de entradas)
		 * Ou seja, pelo metodo HOLDOUT. Nesse problema iremos separar nas proporcoes:
		 * 	- 60% treinamento
		 *  - 20% validacao
		 *  - 20% teste
		 */
		//60% treinamento 20% validacao 20% teste
 		double porcentagem_validacao=0.2;
 		double porcentagem_teste=0.2;
		
		//Calculando o numero de linhas para cada conjunto de dados
 		int num_linhas_treinamento=(int)(entradas.getRowDimension()*(1-porcentagem_validacao-porcentagem_teste));
 		int num_linhas_validacao=(int)(entradas.getRowDimension()*(porcentagem_validacao));
 		int num_linhas_teste=(int)(entradas.getRowDimension()*(porcentagem_teste));
 		//Declarando e instanciando as matrizes que armazenarao os respectivos conjuntos de entradas
 		entradas_treinamento=new Matrix( num_linhas_treinamento, entradas.getColumnDimension() );
		saidas_desejadas_treinamento=new Matrix( num_linhas_treinamento, saidas_desejadas.getColumnDimension());
		entradas_validacao=new Matrix( num_linhas_validacao, entradas.getColumnDimension() );
		saidas_desejadas_validacao=new Matrix( num_linhas_validacao, saidas_desejadas.getColumnDimension());
		entradas_teste=new Matrix( num_linhas_teste, entradas.getColumnDimension() );
		saidas_desejadas_teste=new Matrix( num_linhas_teste, saidas_desejadas.getColumnDimension());
		
		indices_instancias_classe=contar_numero_de_instancias(saidas_desejadas);
		//System.out.println(indices_instancias_classe);
		
 		/*
 		 * Se o metodo de definicao de conjuntos HOLDOUT for do tipo Estratificado,
 		 * eh necessario que cada classe seja representada em numero de instancias
 		 * igualmente no conjunto de treinamento, validacao e teste
 		 */
 		boolean estratificacao=true;
 		if(estratificacao) {
 			System.out.println("------Inicio da Estratificacao-----");
 			
 			//Indice da linha onde sera incluida a nova instancia em cada conjunto
 			int linha_vazia_treinamento=0;
 			int linha_vazia_validacao=0;
 			int linha_vazia_teste=0;
 			
 			/*
 			 * Iteracao sobre as saidas que entraram nos conjuntos
 			 * no laco, eh calculada a proporcao de cada classe em numero de instancia
 			 * que entrara em cada classe, utilizando o Map criado anteriormente para o calculo
 			 */
 			Set<Double> valores_classe = indices_instancias_classe.keySet();
 			Iterator<Double> iterator_valores_classe = valores_classe.iterator();
 			while (iterator_valores_classe.hasNext()) {
 				Double valor=iterator_valores_classe.next();
 				List<Integer> lista_indices = indices_instancias_classe.get(valor);
 				
 				Iterator<Integer> iterator_indices = lista_indices.iterator();
 				//System.out.println("Indices "+lista_indices);

 				int limite_treinamento=(int)Math.ceil((1-porcentagem_teste-porcentagem_validacao)*lista_indices.size());
 				//int limite_teste=(int)Math.ceil( (porcentagem_teste)*lista_indices.size() );
 				int limite_validacao=(int)Math.ceil((porcentagem_validacao)*lista_indices.size());
 				//System.out.println(limite_treinamento);
 				//System.out.println(limite_validacao);
 				//System.out.println(limite_teste);
 				
 				for (int instancia = 0;
 						instancia < limite_treinamento &&
 						linha_vazia_treinamento<entradas_treinamento.getRowDimension();
 						instancia++) {
 					Integer indice = iterator_indices.next();
 					//System.out.println("i="+indice+"->Treinamento");
 					Matrix instancia_entrada=entradas.getMatrix(indice, indice,0, entradas.getColumnDimension()-1);
 					entradas_treinamento.setMatrix(linha_vazia_treinamento, linha_vazia_treinamento,
 							0, entradas.getColumnDimension()-1, instancia_entrada);
 					
 					Matrix instancia_saida=saidas_desejadas.getMatrix(indice, indice,0, saidas_desejadas.getColumnDimension()-1);
 					saidas_desejadas_treinamento.setMatrix(linha_vazia_treinamento, linha_vazia_treinamento,
 							0, saidas_desejadas.getColumnDimension()-1, instancia_saida);
 					
 					linha_vazia_treinamento+=1;
				}
 				
 				for (int instancia = 0;
 						instancia < limite_validacao &&
 						linha_vazia_validacao<entradas_validacao.getRowDimension() &&
 						iterator_indices.hasNext();
 						instancia++) {
 					Integer indice = iterator_indices.next();
 					//System.out.println("i="+indice+"->Validacao");
 					Matrix instancia_entrada=entradas.getMatrix(indice, indice,0, entradas.getColumnDimension()-1);
 					entradas_validacao.setMatrix(linha_vazia_validacao, linha_vazia_validacao,
 							0, entradas.getColumnDimension()-1, instancia_entrada);
 					
 					Matrix instancia_saida=saidas_desejadas.getMatrix(indice, indice,0, saidas_desejadas.getColumnDimension()-1);
 					saidas_desejadas_validacao.setMatrix(linha_vazia_validacao, linha_vazia_validacao,
 							0, saidas_desejadas.getColumnDimension()-1, instancia_saida);
 					
 					linha_vazia_validacao+=1;
				}
 				
 				while (iterator_indices.hasNext()) {
 					Integer indice = iterator_indices.next();
 					//System.out.println("i="+indice+"->Teste");
 					Matrix instancia_entrada=entradas.getMatrix(indice, indice,0, entradas.getColumnDimension()-1);
 					entradas_teste.setMatrix(linha_vazia_teste, linha_vazia_teste,
 							0, entradas.getColumnDimension()-1, instancia_entrada);
 					
 					Matrix instancia_saida=saidas_desejadas.getMatrix(indice, indice,0, saidas_desejadas.getColumnDimension()-1);
 					saidas_desejadas_teste.setMatrix(linha_vazia_teste, linha_vazia_teste,
 							0, saidas_desejadas.getColumnDimension()-1, instancia_saida);
 					
 					linha_vazia_teste+=1;
				}
 				/*
 				double soma_porcentagem_teste=0;
 				while(soma_porcentagem_teste< porcentagem_teste && iterator_indices.hasNext()) {
 					Integer indice = iterator_indices.next();
 					entradas_teste=entradas.getMatrix(indice, indice,0, entradas.getColumnDimension()-1);
 					soma_porcentagem_teste+=porcentagem_por_instancia;
 				}
 				System.out.println("Soma porcentagem teste "+soma_porcentagem_teste);
 				*/
 			}
 			
 			Map<Double,List<Integer>> indices_instancias_classe_treinamento; //Map que armazena em quais indices estao cada uma dos valores de saida desejadas
 			indices_instancias_classe_treinamento=contar_numero_de_instancias(saidas_desejadas_treinamento);
 			Set<Double> classes_treinamento = indices_instancias_classe_treinamento.keySet();
 			Iterator<Double> iterator_classes_treinamento = classes_treinamento.iterator();
 			System.out.println("Teinamento Classe->N_INSTANCIAS");
 			while(iterator_classes_treinamento.hasNext()) {
 				Double valor=iterator_classes_treinamento.next();
 				System.out.print(valor+"->"+indices_instancias_classe_treinamento.get(valor).size()+" ");
 			}
 	 		System.out.println();
 			
 			System.out.println("Estratificacao das classes nos conjuntos:");
 	 		Map<Double,List<Integer>> indices_instancias_classe_validacao; //Map que armazena em quais indices estao cada uma dos valores de saida desejadas
 			indices_instancias_classe_validacao=contar_numero_de_instancias(saidas_desejadas_validacao);
 			Set<Double> classes_validacao = indices_instancias_classe_validacao.keySet();
 			Iterator<Double> iterator_classes_validacao = classes_validacao.iterator();
 			System.out.println("Validacao Classe->N_INSTANCIAS");
 			while(iterator_classes_validacao.hasNext()) {
 				Double valor=iterator_classes_validacao.next();
 				System.out.print(valor+"->"+indices_instancias_classe_validacao.get(valor).size()+" ");
 			}
 			System.out.println();
 	 		
 	 		Map<Double,List<Integer>> indices_instancias_classe_teste; //Map que armazena em quais indices estao cada uma dos valores de saida desejadas
 			indices_instancias_classe_teste=contar_numero_de_instancias(saidas_desejadas_teste);
 			Set<Double> classes_teste = indices_instancias_classe_teste.keySet();
 			Iterator<Double> iterator_classes_teste = classes_teste.iterator();
 			System.out.println("Teste Classe->N_INSTANCIAS");
 			while(iterator_classes_teste.hasNext()) {
 				Double valor=iterator_classes_teste.next();
 				System.out.print(valor+"->"+indices_instancias_classe_teste.get(valor).size()+" ");
 			}
 			System.out.println();
 			
 			System.out.println("------Estratificacao Finalizada-----");
 		}else {
 			//Definicao do conjunto de dados para treinamento
 			int i_inicio_treinamento=0;
 			int i_final_treinamento=i_inicio_treinamento+num_linhas_treinamento-1;
 			entradas_treinamento=entradas.getMatrix(i_inicio_treinamento, i_final_treinamento,
 				0, entradas.getColumnDimension()-1);
 			saidas_desejadas_treinamento=saidas_desejadas.getMatrix(i_inicio_treinamento, i_final_treinamento,
 				0, saidas_desejadas.getColumnDimension()-1);
 			
 			//Definicao do conjunto de dados para validacao
 			int i_inicio_validacao=i_final_treinamento+1;
 			int i_final_validacao=i_inicio_validacao+num_linhas_validacao-1;
 			entradas_validacao=entradas.getMatrix(i_inicio_validacao, i_final_validacao,
 				0, entradas.getColumnDimension()-1) ;
 			saidas_desejadas_validacao=saidas_desejadas.getMatrix(i_inicio_validacao, i_final_validacao,
 				0, saidas_desejadas.getColumnDimension()-1) ;;
 			
 			//Definicao do conjunto de dados para teste
 			int i_inicio_teste=i_final_validacao+1;
 			int i_final_teste=i_inicio_teste+num_linhas_teste-1;
 			entradas_teste=entradas.getMatrix(i_inicio_teste, i_final_teste,
 				0, entradas.getColumnDimension()-1) ;
 			saidas_desejadas_teste=saidas_desejadas.getMatrix(i_inicio_teste, i_final_teste,
 				0, saidas_desejadas.getColumnDimension()-1) ;
 		}
 		
	}
	
	/*
	 * Esse metodo disponibiliza o grafico erro X epocas de uma determinada rede
	 * 		- O numero de epocas eh passado como parametro;
	 */
	public static void grafico_erro_epoca(Rede rede, int numero_limite_epocas) {
		Treinamento treinamento=new Treinamento(rede);
		
		treinamento.treina(entradas_treinamento, saidas_desejadas_treinamento,
				entradas_validacao, saidas_desejadas_validacao, numero_limite_epocas);
	}
	
	//Exibe a matriz de confusao de uma rede, usando os metodos One X One e One X All
	public static void matriz_confusao(Rede rede) {
		
		//Armazena os valores de classes existentes
		Map<Double, List<Integer>> indices_instancias_classe_teste = contar_numero_de_instancias(saidas_desejadas_teste);
		Double[] classes=indices_instancias_classe_teste.keySet().toArray( new Double[0]);

		System.out.println("classes");
		for (int i = 0; i < classes.length; i++) {
			System.out.print(classes[i]+" ");
		}
		System.out.println();
		
		//Estrategia: One X One
		for (int i = 0; i < classes.length; i++) {
			for (int j = i+1; j < classes.length; j++) {
				System.out.println("\nOne x One: "+classes[i]+"x"+classes[j]);
				
				/*
				 * Define as entradas para o One X One: uma nova matriz 
				 * que so contera entradas que tem saida desejadas os valores classes[i] e classes[j]
				 */
				int numero_instancias_classe_i=indices_instancias_classe_teste.get(classes[i]).size(); //numero de instancias da classe[i]
				int numero_instancias_classe_j=indices_instancias_classe_teste.get(classes[j]).size(); //numero de instancias da classe[j]
				int num_linhas_entradas=numero_instancias_classe_i+numero_instancias_classe_j;

				Matrix entradas_one_one=new Matrix(num_linhas_entradas, entradas_teste.getColumnDimension()); //matrix com as instancias da classe[i] e classe[j]
				Matrix saidas_desejadas_one_one=new Matrix(num_linhas_entradas, 1); //matrix com as saidas desejadas das instancias da classe[i] e classe[j]
				int indice_proxima_linha_vazia=0; //indice auxiliar a insercao na matriz de entradas das classes[i] e classes[j]
				
				//inclui as instancias com saidas desejada classe[i] e classe[j] da matriz de entradas separada para teste
				//na matriz destinada para o One x One
				for (int indice_instancia_entradas = 0; indice_instancia_entradas < entradas_teste.getRowDimension(); indice_instancia_entradas++) {
					
					double saida_desejada_instancia=saidas_desejadas_teste.get(indice_instancia_entradas, 0); //saida desejada da instancia do teste
					//a saida desejada eh aquela desejada para o One x One?
					if( saida_desejada_instancia==classes[i] ||	saida_desejada_instancia==classes[j]) {
						Matrix entrada_instancia=entradas_teste.getMatrix(indice_instancia_entradas, indice_instancia_entradas, 0, entradas_teste.getColumnDimension()-1);
						entradas_one_one.setMatrix(indice_proxima_linha_vazia, indice_proxima_linha_vazia,
													0, entradas_one_one.getColumnDimension()-1,
													entrada_instancia);
						saidas_desejadas_one_one.set(indice_proxima_linha_vazia,0,saida_desejada_instancia);
						
						indice_proxima_linha_vazia+=1;
					}
				}
				
				System.out.println("Entradas Teste");
				entradas_teste.print(entradas_teste.getColumnDimension(), 3);
				System.out.println("Saidas Desejadas Teste");
				saidas_desejadas_teste.print(saidas_desejadas_teste.getColumnDimension(), 3);
				System.out.println("Entradas One x One");
				entradas_one_one.print(entradas_one_one.getColumnDimension(), 3);
				System.out.println("Saidas One x One");
				saidas_desejadas_one_one.print(saidas_desejadas_one_one.getColumnDimension(), 3);
				
				//rede.set_problema(entradas_one_one, saidas_desejadas_one_one);
				//Matrix saidas=rede.get_saidas();
				
				int falso_negativo=0;
				int falso_positivo=0;
				int verdadeiro_positivo=0;
				int verdadeiro_negativo=0;
				
				//for (int k = 0; k < saidas.getRowDimension(); k++) {
					
				//}
				System.out.println("----------Fim One x One: "+classes[i]+"x"+classes[j]+"--------\n");
			}
		}
	}
	
	/*
	 * Esse metodo devolve uma lista que contem o numero de valores repetidos
	 * de cada valor dentro de uma matriz de dados passada como argumento
	 */
	public static Map<Double,List<Integer>> contar_numero_de_instancias(Matrix dados) {
		Map<Double,List<Integer>> indice_de_instancias=new HashMap<Double,List<Integer>>();
		for (int i = 0; i < dados.getRowDimension(); i++) {
			Double valor=dados.get(i, 0);
			List<Integer> num_indice_intancia = indice_de_instancias.get(valor);
			if(num_indice_intancia==null) num_indice_intancia=new ArrayList<Integer>();
			num_indice_intancia.add(i);
			indice_de_instancias.put(valor,num_indice_intancia);
		}
		return indice_de_instancias;
	}

	public static Matrix adiciona_bias(Matrix entradas_sem_bias) {
		Matrix entradas;
		//Adicionando o Bias
		entradas=new Matrix( entradas_sem_bias.getRowDimension(), entradas_sem_bias.getColumnDimension()+1 );
		for (int i = 0; i < entradas_sem_bias.getRowDimension(); i++) {
			Matrix entrada_sem_bias=entradas_sem_bias.getMatrix(i, i, 0, entradas_sem_bias.getColumnDimension()-1);
			entradas.setMatrix(i, i, 0, entradas.getColumnDimension()-2, entrada_sem_bias);
			//Acrescenta 1 como bias
			double bias=1;
			entradas.set(i, entradas.getColumnDimension()-1, bias);
		}
		return entradas;
	}
}
