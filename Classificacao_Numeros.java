import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.jfree.ui.RefineryUtilities;
import Jama.Matrix;


public class Classificacao_Numeros {
	//static Matrix entradas; //conjunto de entradas com bias que contem todas as instancias do arquivo
	//static Matrix saidas_desejadas; //conjunto de saidas desejadas que contem todas as saidas da instancia do arquivo
	
	Matrix entradas_treinamento; //conjunto de entradas para treinamento retirado do conjunto de entradas
	Matrix saidas_desejadas_treinamento; //conjunto de saidas desejadas para as intancias do conjunto de treinamento retirado do conjunto de saidas desejadas
	Matrix entradas_validacao; //conjunto de entradas para validaca retirado do conjunto de entradas
	Matrix saidas_desejadas_validacao; //conjunto de saidas desejadas para as intancias do conjunto de validaca retirado do conjunto de saidas desejadas
	Matrix entradas_teste; //conjunto de entradas para validacao retirado do conjunto de entradas
	Matrix saidas_desejadas_teste; //conjunto de saidas desejadas para as intancias do conjunto de teste retirado do conjunto de saidas desejadas
	
	Map<Double,List<Integer>> indices_instancias_classe; //Map que armazena em quais indices estao cada uma dos valores de saida desejadas
	static boolean pesos_aleatorios;
	
	public Classificacao_Numeros(Matrix[][] conjuntos_dados ) {
		this.entradas_treinamento=conjuntos_dados[0][0];
		this.entradas_validacao=conjuntos_dados[1][0];
		this.entradas_teste=conjuntos_dados[2][0];
		this.saidas_desejadas_treinamento=conjuntos_dados[0][1];
		this.saidas_desejadas_validacao=conjuntos_dados[1][1];
		this.saidas_desejadas_teste=conjuntos_dados[2][1];
		
		indices_instancias_classe=Holdout.contar_numero_de_instancias(saidas_desejadas_treinamento);
	}
	
	/* -----------------------MLP----------------------------------------------
	 * Uma nova rede MLP eh criada com numero de neuronios e taxa de aprendizado (parametros).
	 * Na instanciacao do novo objeto se define tambem se a taxa de aprendizado
	 * ira ser atualizada durante o treinamento ou nao. Alem disso, o modo de treinamento (padrao a padrao=1 
	 * ou batelada=2) da rede tbm sera pasado como parametro do metodo.
	 * O numero_epocas eh a quantidade de epocas em que a rede será treinada
	 */
	public void analisa_mlp(double taxa_aprendizado_inicial, boolean taxa_aprendizado_variavel, boolean pesos_aleatorios,
							int numero_neuronios_escondidos, int modo_treinamento, int numero_epocas) {
		
		Rede mlp=new MLP(numero_neuronios_escondidos, taxa_aprendizado_inicial, taxa_aprendizado_variavel);
		mlp.set_modo_treinamento(modo_treinamento); //Configura a rede para fazer o treinamento
		
		System.out.println("\t#--------------Inicio da Fase de Treinamento-------#");
		//Treinar MLP
		Treinamento treinamento=new Treinamento(mlp);
		Matrix erros_epocas =treinamento.treina(adiciona_bias(entradas_treinamento), saidas_desejadas_treinamento,
									adiciona_bias(entradas_validacao), saidas_desejadas_validacao,
									numero_epocas, pesos_aleatorios);
		erros_epocas.print(erros_epocas.getColumnDimension(), 3);
		System.out.println("\t#--------------Termino da Fase de Treinamento--------#");
		
		System.out.println("\t#----------------Inicio da Exibicao do Grafico------------------#");
		//Mostrar grafico Epoca X Erro Total
		grafico_erro_epoca(erros_epocas);
		System.out.println("\t#----------------Termino da Exibicao do Grafico-----------------#");
		
		System.out.println("\t#----------------Inicio da Matriz de Confusao------------------#");
		//Montar matrz de confusao com MLP treinada na analise
		matriz_confusao(mlp); //Calcula as matries de confusao para a MLP
		System.out.println("\t#----------------Termino da Matriz de Confusao------------------#");
	}
	
	/* -----------------------LVQ----------------------------------------------
	 * Uma nova LVQ eh criada com numero de neuronios por classe e taxa de aprendizado inicial (parametros).
	 * Na instanciacao do novo objeto eh passado tambem quantas instancias existem por classe.
	 * O numero_epocas eh a quantidade de epocas em que a rede será treinada
	 */
	public void analisa_lvq(double taxa_aprendizado_inicial, boolean taxa_aprendizado_variavel, boolean pesos_aleatorios,
							int numero_neuronios_classe, int numero_epocas) {
		double[] classes; //array das classes existentes no conjunto de saidas desejadas (rotulos da LVQ)
		Set<Double> classes_keySet = indices_instancias_classe.keySet();
		classes=new double[classes_keySet.size()];
		Iterator<Double> iterator_classes = classes_keySet.iterator();
		for(int i=0; iterator_classes.hasNext();i++) { 
			classes[i]=iterator_classes.next();
		}
		Rede lvq=new LVQ(numero_neuronios_classe, taxa_aprendizado_inicial, classes);
		
		System.out.println("\t#--------------Inicio da Fase de Treinamento---------------#");
		System.out.println("\t\tNumero de limite de epocas="+numero_epocas);
		//Treinar LVQ
		Treinamento treinamento=new Treinamento(lvq);
		
		Matrix erros_epocas =treinamento.treina(entradas_treinamento, saidas_desejadas_treinamento,
									entradas_validacao, saidas_desejadas_validacao,
									numero_epocas, pesos_aleatorios);
		System.out.println("\t#--------------Termino da Fase de Treinamento---------------#");
		
		System.out.println("\n#----------------Inicio da Matriz de Confusao------------------#");
		matriz_confusao(lvq); //Calcula as matries de confusao para a LVQ
		System.out.println("#----------------Termino da Matriz de Confusao------------------#");
	}
	
	/*
	 * Esse metodo disponibiliza o grafico erro X epocas de uma determinada rede
	 * 		- O numero de epocas eh passado como parametro;
	 */
	public static void grafico_erro_epoca(Matrix erros_epocas) {
		Grafico grafico = new Grafico("Erro x Epoca", erros_epocas.transpose());
        grafico.pack();
        RefineryUtilities.centerFrameOnScreen(grafico);
        grafico.setVisible(true);
	}
	
	//Exibe a matriz de confusao de uma rede, usando os metodos One X One e One X All
	public void matriz_confusao(Rede rede) {
		//Armazena os valores de classes existentes
		Map<Double, List<Integer>> indices_instancias_classe_teste = Holdout.contar_numero_de_instancias(this.saidas_desejadas_teste);
		Double[] classes=indices_instancias_classe_teste.keySet().toArray( new Double[0]);

		System.out.println("Classes no conjunto de Teste: ");
		for (int i = 0; i < classes.length; i++) {
			System.out.print(classes[i]+", ");
		}
		System.out.println(".");
		
		//Estrategia: One X One
		for (int i = 0; i < classes.length; i++) {
			for (int j = i+1; j < classes.length; j++) {
				System.out.println("\n-----------One x One: "+classes[i]+"x"+classes[j]+"------");
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
				
				//System.out.println("Entradas Teste");
				//entradas_teste.print(entradas_teste.getColumnDimension(), 3);
				//System.out.println("Saidas Desejadas Teste");
				//saidas_desejadas_teste.print(saidas_desejadas_teste.getColumnDimension(), 3);
				
				//Elementos da matriz de confusao
				int falso_negativo=0;
				int falso_positivo=0;
				int verdadeiro_positivo=0;
				int verdadeiro_negativo=0;
				
				try {
					LVQ lvq=(LVQ)rede; //Caso a rede seja uma MLP, uma excessao que o cast nao eh possivel eh lancada
				}catch(ClassCastException cce){
					entradas_one_one=adiciona_bias(entradas_one_one);
				}
				
				//Rodar rede com as entradas referentes ao One X One e armazenar as saidas
				rede.set_problema(entradas_one_one, saidas_desejadas_one_one);
				Matrix saidas=rede.get_saidas();
				
				//Dadas as saidas da rede, compara-las com as saidas desejadas para montar
				//a matriz de confusao
				for (int k = 0; k < saidas_desejadas_one_one.getRowDimension(); k++) {
					if( saidas_desejadas_one_one.get(k, 0)==classes[i] ) { //Se a classe real for classes[i]
						//Se a classe predita for igual a classe real
						//Eh necessario fazer a quantizacao para realizar a comparacao, dado que o resultado nunca sera exato.
						//Eh preciso tomar cuidado com os indices do array, por isso os ifs aninhados.
						if(k+1<saidas_desejadas_one_one.getRowDimension()){
							if(saidas.get(k, 0) >= saidas_desejadas_one_one.get(k, 0) && saidas.get(k, 0) < saidas_desejadas_one_one.get(k+1, 0)) {
								verdadeiro_positivo+=1;
							}else {	//Se a classe predita for diferente da classe real
								falso_negativo+=1;
							}	
						}else{
							if(saidas.get(k, 0) >= saidas_desejadas_one_one.get(k, 0)) {
								verdadeiro_positivo+=1;
							}else {	//Se a classe predita for diferente da classe real
								falso_negativo+=1;
							}
						}
					}else { //Se a classe real for clases[j]
						if(k+1<saidas_desejadas_one_one.getRowDimension()){
							if(saidas.get(k, 0) >= saidas_desejadas_one_one.get(k, 0) && saidas.get(k, 0) < saidas_desejadas_one_one.get(k+1, 0)) { //Se a classe predita for igual a classe real
								verdadeiro_negativo+=1;
							}else { //Se a classe predita for diferente da classe real
								falso_positivo+=1;
							}
						}else{
							if(saidas.get(k, 0) >= saidas_desejadas_one_one.get(k, 0)) { //Se a classe predita for igual a classe real
								verdadeiro_negativo+=1;
							}else { //Se a classe predita for diferente da classe real
								falso_positivo+=1;
							}
						}
					}
				}
				System.out.println("Matriz de confusao" +classes[i]+" X "+classes[j]);
				System.out.println("Verdadeiro positivo: " + verdadeiro_positivo);
				System.out.println("Falso negativo: "+falso_negativo);
				System.out.println("Falso positivo: "+falso_positivo);
				System.out.println("Verdadeiro negativo: "+verdadeiro_negativo);
				
				/*TODO contabiliza cada um dos elementos da matriz de confusao
				//Medidas extraidas da matriz de confusao
				double sensibilidade=verdadeiro_positivo/(verdadeiro_positivo+falso_negativo); //taxa de verdadeiros positivos ou revocacao
				double taxa_falsos_positivos=falso_positivo/(verdadeiro_negativo+falso_positivo);
				double especificidade=verdadeiro_negativo/(falso_positivo+falso_positivo); //taxa de verdadeiros negativos
				double precisao= verdadeiro_positivo/(verdadeiro_positivo+verdadeiro_negativo);
				double preditividade_negativa=taxa_falsos_positivos/(taxa_falsos_positivos+falso_negativo);
				double taxa_falsas_descobertas=falso_positivo/(verdadeiro_positivo+falso_positivo);
				double taxa_acuracia=(verdadeiro_negativo+verdadeiro_positivo)/(falso_negativo+falso_positivo+verdadeiro_negativo+verdadeiro_positivo);
				double taxa_erro=(falso_negativo+falso_positivo)/(falso_negativo+falso_positivo+verdadeiro_negativo+verdadeiro_positivo);
				
				System.out.println("\nSensibilidade="+sensibilidade);
				System.out.println("\nTaxa de falsos positivos="+taxa_falsos_positivos);
				System.out.println("\nEspecificidade="+especificidade);
				System.out.println("\nPrecisao="+precisao);
				System.out.println("\nPreditividade Negativa="+preditividade_negativa);
				System.out.println("\nTaxa de falsas descobertas="+taxa_falsas_descobertas);
				System.out.println("\nTaxa de Acuracidade="+taxa_acuracia);
				System.out.println("\nTaxa de Erro="+taxa_erro);
				*/
				System.out.println("----------Fim One x One: "+classes[i]+"x"+classes[j]+"--------\n");
				
			}
		}
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
/*-----------Testando MLP--------------
Matrix pesos_a= new Matrix( mlp.numero_neuronios, entradas_teste.getColumnDimension()+1 );
Matrix pesos_b= new Matrix( saidas_desejadas_teste.getColumnDimension(), mlp.numero_neuronios+1 );
Treinamento.gera_pesos_aleatorios(pesos_a);
Treinamento.gera_pesos_aleatorios(pesos_b);

Matrix entradas_teste_bias=adiciona_bias(entradas_teste);

//entradas_teste_bias.print(entradas_teste_bias.getColumnDimension(), 3);
//saidas_desejadas_teste.print(saidas_desejadas_teste.getColumnDimension(), 3);
//pesos_a.print(pesos_a.getColumnDimension(), 3);
//pesos_b.print(pesos_b.getColumnDimension(), 3);

mlp.set_pesos(pesos_a, pesos_b);
mlp.set_problema(entradas_teste_bias, saidas_desejadas_teste);
double erro = mlp.get_erro();
System.out.println("Erro="+erro);
Matrix saidas_epoca_mlp=mlp.get_saidas();

System.out.println("Saida epoca Mlp");
saidas_epoca_mlp.print(saidas_epoca_mlp.getColumnDimension(), 3);
-------------------------*/
/*-----------Testando LVQ--------------
pesos_a= new Matrix( lvq.numero_neuronios, entradas_treinamento.getColumnDimension() );
pesos_b= new Matrix( saidas_desejadas_treinamento.getColumnDimension(), lvq.numero_neuronios );
Treinamento.gera_pesos_aleatorios(pesos_a);
Treinamento.gera_pesos_aleatorios(pesos_b);

lvq.set_necessidade_atualizacao();
lvq.set_pesos(pesos_a, pesos_b);
lvq.set_problema(entradas_treinamento, saidas_desejadas_treinamento);
lvq.set_necessidade_atualizacao();
Matrix saidas_epoca_lvq=lvq.get_saidas();
-------------------------------*/