import Jama.Matrix;


public class Principal {
	public static void main(String[] args) {
		/* args[0] = Nome do arquivo do conjunto de dados de treino
		 * args[1] = Nome do arquivo do conjunto de dados de validação
		 * args[2] = Nome do arquivo do conjunto de dados de teste
		 * args[3] = Taxa de aprendizado inicial
		 * args[4] = Número de neurônios de camada escondida (MLP)
		 * args[5] = Número de neurônios para a camada classe (LVQ)
		 * args[6] = Inicialização de pesos (zero/aleatório)
		 * */
		//Parametros que deve ser passados como parametro
		String nome_arquivo_conjunto_dados; //Nome do conjunto de dados
		String nome_arquivo_dados_treinamento = null;
		String nome_arquivo_dados_validacao = null;
		String nome_arquivo_dados_teste = null;
		//nome_arquivo_conjunto_dados="conjunto_dados.txt";
		nome_arquivo_conjunto_dados="optdigits.total.txt";
		double taxa_aprendizado_inicial=0.1;
		boolean taxa_aprendizado_variavel=true;
		boolean pesos_aleatorios=true;
		//Para MLP
		int numero_neuronios_escondidos=10;
		int modo_treinamento=1; //padrao a padrao=1 e batelada=2
		//Para LVQ
		int numero_neuronios_classe=3;
		//Numero maximo de epocas para analise
		int numero_epocas=50;
		
		if(args.length != 7 && args.length != 5){
			System.out.println("Favor inserir os seguintes dados ao chamar o programa, na ordem especifica listada a seguir:"
					+ " \n"
					+ "Nome do arquivo do conjunto de dados de treino \n"
					+ "Nome do arquivo do conjunto de dados de validação \n"
					+ "Nome do arquivo do conjunto de dados de teste \n"
					+ "Taxa de aprendizado inicial \n"
					+ "Número de neurônios de camada escondida (MLP) \n"
					+ "Número de neurônios para a camada classe (LVQ) \n"
					+ "Inicialização de pesos (zero/aleatório) (inserir z ou a) \n");
			System.out.println("\n\n OU:"
					+ " \n"
					+ "Nome do arquivo com todos os dados \n"
					+ "Taxa de aprendizado inicial \n"
					+ "Número de neurônios de camada escondida (MLP) \n"
					+ "Número de neurônios para a camada classe (LVQ) \n"
					+ "Inicialização de pesos (zero/aleatório) (inserir z ou a) \n");
			System.exit(0);
		}
		
		if(args.length==7){
			nome_arquivo_dados_treinamento = args[0];
			nome_arquivo_dados_validacao = args[1];
			nome_arquivo_dados_teste = args[2];
			
			taxa_aprendizado_inicial = Double.parseDouble(args[3]);
			numero_neuronios_escondidos = Integer.parseInt(args[4]);
			numero_neuronios_classe = Integer.parseInt(args[5]);
			if(args[6].equalsIgnoreCase("z")){
				pesos_aleatorios = false;
			}else if(args[6].equalsIgnoreCase("a")){
				pesos_aleatorios = true;
			}else{
				System.out.println("Erro ao inserir dados.");
				System.exit(0);
			}	
		}else if(args.length==5){
			nome_arquivo_conjunto_dados = args[0];
			taxa_aprendizado_inicial = Double.parseDouble(args[1]);
			numero_neuronios_escondidos = Integer.parseInt(args[2]);
			numero_neuronios_classe = Integer.parseInt(args[3]);
			if(args[4].equalsIgnoreCase("z")){
				pesos_aleatorios = false;
			}else if(args[4].equalsIgnoreCase("a")){
				pesos_aleatorios = true;
			}else{
				System.out.println("Erro ao inserir dados.");
				System.exit(0);
			}
		}
		
		System.out.println("\n#-----------Lendo Arquivo de Entrada------------------#");
		/*
		 * Le o arquivo do conjunto de dados e separa em:
		 * 	- Atributos (Entradas); e
		 *  - Atributos Classe (Saidas Desejadas).
		 * Colocando-os em matrizes
		 */
		Classificacao_Numeros classificacao_numeros = null;
		if(args.length==7){
			Situacao_Problema situacao_problema_conjunto_dados_treinamento = Leitura_Arquivo.obtem_dados(nome_arquivo_dados_treinamento);
			
			Matrix entradas_treinamento=situacao_problema_conjunto_dados_treinamento.get_entrada();
			//System.out.println("Antes: "+entradas_treinamento.getColumnDimension()+" X "+entradas_treinamento.getRowDimension());
			entradas_treinamento = Pre_Processamento.normaliza_sigmoidal(entradas_treinamento);
			//System.out.println("Depois: "+entradas_treinamento.getColumnDimension()+" X "+entradas_treinamento.getRowDimension());
			
			Matrix saidas_desejadas_treinamento = situacao_problema_conjunto_dados_treinamento.get_saida();
			//System.out.println("Antes: "+saidas_desejadas_treinamento.getColumnDimension()+" X "+saidas_desejadas_treinamento.getRowDimension());
			saidas_desejadas_treinamento = Pre_Processamento.normaliza_minmax(saidas_desejadas_treinamento);
			//System.out.println("Depois: "+saidas_desejadas_treinamento.getColumnDimension()+" X "+saidas_desejadas_treinamento.getRowDimension());
						
			Situacao_Problema situacao_problema_conjunto_dados_validacao = Leitura_Arquivo.obtem_dados(nome_arquivo_dados_validacao);
			
			Matrix entradas_validacao = situacao_problema_conjunto_dados_validacao.get_entrada();
			//System.out.println("Antes: "+entradas_validacao.getColumnDimension()+" X "+entradas_validacao.getRowDimension());
			entradas_validacao = Pre_Processamento.normaliza_sigmoidal(entradas_validacao);
			//System.out.println("Depois: "+entradas_validacao.getColumnDimension()+" X "+entradas_validacao.getRowDimension());
			
			Matrix saidas_desejadas_validacao=situacao_problema_conjunto_dados_validacao.get_saida();
			//System.out.println("Antes: "+saidas_desejadas_validacao.getColumnDimension()+" X "+saidas_desejadas_validacao.getRowDimension());
			saidas_desejadas_validacao = Pre_Processamento.normaliza_minmax(saidas_desejadas_validacao);
			//System.out.println("Depois: "+saidas_desejadas_validacao.getColumnDimension()+" X "+saidas_desejadas_validacao.getRowDimension());
			
			Situacao_Problema situacao_problema_conjunto_dados_teste = Leitura_Arquivo.obtem_dados(nome_arquivo_dados_teste);

			Matrix entradas_teste=situacao_problema_conjunto_dados_teste.get_entrada();
			//System.out.println("Antes: "+entradas_teste.getColumnDimension()+" X "+entradas_teste.getRowDimension());
			entradas_teste = Pre_Processamento.normaliza_sigmoidal(entradas_teste);
			//System.out.println("Depois: "+entradas_teste.getColumnDimension()+" X "+entradas_teste.getRowDimension());
			
			Matrix saidas_desejadas_teste=situacao_problema_conjunto_dados_teste.get_saida();
			//System.out.println("Antes: "+saidas_desejadas_teste.getColumnDimension()+" X "+saidas_desejadas_teste.getRowDimension());
			saidas_desejadas_teste = Pre_Processamento.normaliza_minmax(saidas_desejadas_teste);
			//System.out.println("Depois: "+saidas_desejadas_teste.getColumnDimension()+" X "+saidas_desejadas_teste.getRowDimension());
			
			System.out.println("#-----------Termino da Leitura Arquivo de Entrada-----#");
			
			System.out.println("\n#-----------Inicio da Separacao dos Conjuntos--------------#");
			Matrix[][] conjuntos_dados = new Matrix[3][2];
			
			conjuntos_dados[0][0]=entradas_treinamento;
			conjuntos_dados[1][0]=entradas_validacao;
			conjuntos_dados[2][0]=entradas_teste;
			conjuntos_dados[0][1]=saidas_desejadas_treinamento;
			conjuntos_dados[1][1]=saidas_desejadas_validacao;
			conjuntos_dados[2][1]=saidas_desejadas_teste;
			System.out.println("#-----------Termino da Separacao dos Conjuntos------------------");
			
			classificacao_numeros = new Classificacao_Numeros(conjuntos_dados);
		}else if(args.length==5){
			Situacao_Problema situacao_problema_conjunto_dados = Leitura_Arquivo.obtem_dados(nome_arquivo_conjunto_dados);
			Matrix entradas=situacao_problema_conjunto_dados.get_entrada();
			//System.out.println("Comeca normalizacao entradas.\n");
			entradas = Pre_Processamento.remove_desvio_baixo(entradas, 2);
			//entradas.print(10, 5);
			//System.exit(0);
			entradas = Pre_Processamento.normaliza_zscore(entradas);
			//System.out.println("Termina normalizacao entradas.\n");

			Matrix saidas_desejadas=situacao_problema_conjunto_dados.get_saida();
			//System.out.println("Comeca normalizacao saida.\n");
			saidas_desejadas = Pre_Processamento.normaliza_minmax(saidas_desejadas);
			//System.out.println("Termina normalizacao saidas.\n");
			System.out.println("#-----------Termino da Leitura Arquivo de Entrada-----#");
			
			System.out.println("\n#-----------Inicio da Separacao dos Conjuntos--------------#");
			boolean estratificado=true;
			Holdout holdout=new Holdout();
			Matrix[][] conjuntos_dados=holdout.separa_conjunto(entradas, saidas_desejadas, estratificado);
			System.out.println("#-----------Termino da Separacao dos Conjuntos------------------");
			
			classificacao_numeros = new Classificacao_Numeros(conjuntos_dados);
		}
		
		
		System.out.println("\n#----------------Inicio da Analise da MLP------------------#");
		classificacao_numeros.analisa_mlp(taxa_aprendizado_inicial, taxa_aprendizado_variavel, pesos_aleatorios,
										numero_neuronios_escondidos, modo_treinamento, numero_epocas);
		System.out.println("#----------------Termino da Analise da MLP----------------#");
		
		/*
		System.out.println("\n#----------------Inicio da Analise da LVQ------------------#");
		classificacao_numeros.analisa_lvq(taxa_aprendizado_inicial, taxa_aprendizado_variavel, pesos_aleatorios,
										numero_neuronios_classe, numero_epocas);
		System.out.println("#----------------Termino da Analise da LVQ----------------#");
		*/
	}
}
