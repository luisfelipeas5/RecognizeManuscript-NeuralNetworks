import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;


public class Holdout {

	/*
	 * Esse metodo retorna uma matriz de conjunto de dados:
	 * As linhas: 
	 *  - a primeira linha se refere ao conjunto treinamento
	 *  - a segunda linha se refere ao conjunto validacao
	 *  - a terceira linha se refere ao conjunto de teste
	 * As colunas:
	 * 	- a primeira coluna se refere as entradas de cada conjunto
	 * 	- a segunda coluna se refere as saidas desejadas de cada conjunto
	 * Recebe como parametro o connjunto de entrada, o conjunto de saidas desejadas
	 * e uma estrutura que reune os valores de classe existentes no conjunto e o indice de cada instancia
	 * de cada classe
	 */
	public Matrix[][] separa_conjunto(Matrix entradas, Matrix saidas_desejadas,
			boolean estratificado) {
		//Armazena os valores de classes existentes
		Map<Double, List<Integer>> indices_instancias_classe = contar_numero_de_instancias(saidas_desejadas);
		
		Matrix[][] conjuntos_dados=new Matrix[3][3];
		Matrix entradas_treinamento, saidas_desejadas_treinamento;
		Matrix entradas_validacao, saidas_desejadas_validacao;
		Matrix entradas_teste, saidas_desejadas_teste;
		/*
		 * Troca as posicoes das instancias para que o holdout 
		 * separe randomicamente as instancias nos conjuntos
		 */
		embaralhar_conjuntos(entradas, saidas_desejadas);
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
		int num_linhas_treinamento=(int)Math.ceil(entradas.getRowDimension()*(1-porcentagem_validacao-porcentagem_teste));
		int num_linhas_validacao=(int)Math.ceil(entradas.getRowDimension()*(porcentagem_validacao));
		int num_linhas_teste=(int)Math.ceil(entradas.getRowDimension()*(porcentagem_teste));
		//Declarando e instanciando as matrizes que armazenarao os respectivos conjuntos de entradas
		entradas_treinamento=new Matrix( num_linhas_treinamento, entradas.getColumnDimension() );
		saidas_desejadas_treinamento=new Matrix( num_linhas_treinamento, saidas_desejadas.getColumnDimension());
		entradas_validacao=new Matrix( num_linhas_validacao, entradas.getColumnDimension() );
		saidas_desejadas_validacao=new Matrix( num_linhas_validacao, saidas_desejadas.getColumnDimension());
		entradas_teste=new Matrix( num_linhas_teste, entradas.getColumnDimension() );
		saidas_desejadas_teste=new Matrix( num_linhas_teste, saidas_desejadas.getColumnDimension());
		
		/*
		 * Se o metodo de definicao de conjuntos HOLDOUT for do tipo Estratificado,
		 * eh necessario que cada classe seja representada em numero de instancias
		 * igualmente no conjunto de treinamento, validacao e teste
		 */
		if(estratificado) {
			System.out.println("\t#----------Inicio da Estratificacao--------#");
			
			//Indice da linha onde sera incluida a nova instancia em cada conjunto
			int linha_vazia_treinamento=0;
			int linha_vazia_validacao=0;
			int linha_vazia_teste=0;
			
			/*
			 * Iteracao sobre as saidas que entraram nos conjuntos
			 * no laco, eh calculada a proporcao de cada classe em numero de instancia
			 * que entrara em cada classe, utilizando o Map criado anteriormente para o calculo
			 */
			Set<Double> valores_classe = indices_instancias_classe.keySet();//valores das classes existentes
			Iterator<Double> iterator_valores_classe = valores_classe.iterator();
			//Itera sobre todos os valores de classes
			while (iterator_valores_classe.hasNext()) {
				Double valor=iterator_valores_classe.next(); //valor da classe iterada
				//System.out.println("Classe "+valor);
				
				//Indices onde estao as instancias da classe iterada
				List<Integer> lista_indices = indices_instancias_classe.get(valor);
				Iterator<Integer> iterator_indices = lista_indices.iterator();
				//System.out.println("\tIndices da classe iterada"+lista_indices);

				//Numero de instancias da classe iterada que cada conjunto ira receber
				int limite_treinamento=(int)Math.ceil((1-porcentagem_teste-porcentagem_validacao)*lista_indices.size());
				int limite_validacao=(int)Math.ceil((porcentagem_validacao)*lista_indices.size());
				//int limite_teste=(int)Math.ceil( (porcentagem_teste)*lista_indices.size() );
				
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
			
			System.out.println("\t#------Estratificacao Finalizada-----#");
		}else { //Caso nao seja estratificado
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
		conjuntos_dados[0][0]=entradas_treinamento;
		conjuntos_dados[1][0]=entradas_validacao;
		conjuntos_dados[2][0]=entradas_teste;
		conjuntos_dados[0][1]=saidas_desejadas_treinamento;
		conjuntos_dados[1][1]=saidas_desejadas_validacao;
		conjuntos_dados[2][1]=saidas_desejadas_teste;
		
		return conjuntos_dados;
	}

	/*
	 * Dado duas matrizes, o metodo troca as instancias de lugares para
	 * embaralhar o conjunto de dados originais, utilizando um metodo Randomico
	 */
	public static void embaralhar_conjuntos(Matrix entradas, Matrix saidas_desejadas) {
		Random random=new Random();
		for(int indice_instancia=0; indice_instancia<entradas.getRowDimension(); indice_instancia++) {
			int nova_indice_instancia=random.nextInt(entradas.getRowDimension());
			
			Matrix entrada=entradas.getMatrix(indice_instancia, indice_instancia, 0, entradas.getColumnDimension()-1);
			Matrix saida_desejada=saidas_desejadas.getMatrix(indice_instancia, indice_instancia, 0, saidas_desejadas.getColumnDimension()-1);
			
			Matrix nova_entrada=entradas.getMatrix(nova_indice_instancia, nova_indice_instancia, 0, entradas.getColumnDimension()-1);
			Matrix nova_saida_desejada=saidas_desejadas.getMatrix(nova_indice_instancia, nova_indice_instancia, 0, saidas_desejadas.getColumnDimension()-1);
			
			entradas.setMatrix(indice_instancia, indice_instancia, 0, entradas.getColumnDimension()-1, nova_entrada);
			entradas.setMatrix(nova_indice_instancia, nova_indice_instancia, 0, entradas.getColumnDimension()-1, entrada);
			saidas_desejadas.setMatrix(indice_instancia, indice_instancia, 0, saidas_desejadas.getColumnDimension()-1, nova_saida_desejada);
			saidas_desejadas.setMatrix(nova_indice_instancia, nova_indice_instancia, 0, saidas_desejadas.getColumnDimension()-1, saida_desejada);
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
}
