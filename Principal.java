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
		String nome_arquivo_conjunto_dados = null; //Nome do conjunto de dados
		String nome_arquivo_dados_treinamento = null;
		String nome_arquivo_dados_validacao = null;
		String nome_arquivo_dados_teste = null;
		String tipo_normalizacao = "nenhum";
		String tipo_remocao = "nenhum";
		double valor_remocao = 0.0;
		double taxa_aprendizado_inicial = 0.1;
		boolean taxa_aprendizado_variavel = true;
		boolean pesos_aleatorios = false;
		double intervalo_pesos_aleatorios = 0.5;		
		//Numero maximo de epocas e erro maximo para analise
		int numero_epocas = 100;
		double limiar_erro = 0.00;

		//Para MLP
		int numero_neuronios_escondidos = 5;
		int modo_treinamento = 1; //padrao a padrao=1 e batelada=2
		
		//Para LVQ
		int numero_neuronios_classe = 3;
		
		
		if(args.length != 5 && args.length != 7 && args.length != 11){
			System.out.println("Por favor utilizar um dos seguintes padroes de passagem de parametros ao executar esse programa:"
					+ "\n\n"
					+ "Entrada padrao: \n\n"
					+ "1. Nome do arquivo do conjunto de dados de treino. \n\n"
					+ "2. Nome do arquivo do conjunto de dados de validacao. \n\n"
					+ "3. Nome do arquivo do conjunto de dados de teste. \n\n"
					+ "4. Taxa de aprendizado inicial. \n\n"
					+ "5. Numero de neuronios de camada escondida (MLP). \n\n"
					+ "6. Numero de neuronios para a camada classe (LVQ). \n\n"
					+ "7. Inicializacao de pesos:\n"
					+ "  -> Inserir 'z' para matriz de pesos 0 ou 'a' para matriz de pesos aleatorios. \n\n"
					+ "OU:\n\n"
					+ "Entrada padrao modificada: \n"
					+ "1. Nome do arquivo com todos os dados \n\n"
					+ "2. Taxa de aprendizado inicial \n\n"
					+ "3. Numero de neuronios de camada escondida (MLP) \n\n"
					+ "4. Numero de neuronios para a camada classe (LVQ) \n\n"
					+ "5. Inicializacao de pesos:\n"
					+ "  -> Inserir 'z' para matriz de pesos 0 ou 'a' para matriz de pesos aleatorios.\n\n"
					+ "OU:\n\n"
					+ "Entrada extendida: \n"
					+ "1. Nome do arquivo com todos os dados. \n\n"
					+ "  -> Inserir: 'nenhum' ou 'max' ou 'minmax' ou 'zscore' ou 'sigmoidal'. \n\n"
					+ "2. Tipo de reducao de dados: remover colunas preenchidas com zeros ou colunas com baixo desvio padrao: \n"
					+ "  -> Inserir: 'nenhum' ou 'zeros' ou 'desvio'. \n\n"
					+ "3. Valor de corte: \n"
					+ "  3.1. Em caso de eleminacao de colunas por atributos com zeros: \n"
					+ "  -> Inserir valor entre 0 e 100, correspondente a porcentagem de zeros que um atributo "
					+ "deve ter para ser eliminado. \n"
					+ "  3.2. Em caso de eliminacao de colunas por atributos com baixo desvio padrao: \n"
					+ "  -> Inserir valor minimo de desvio padrao desejado para as colunas. "
					+ "Colunas que com desvio padrao menor ou igual ao inserido serao eliminadas. \n"
					+ "  3.3. Em caso do paramtro 3 = 'nenhum': \n"
					+ "  -> Inserir: 0. \n\n"
					+ "4. Tipo de normalizacao: \n "
					+ "5. Taxa de aprendizado inicial. \n\n"
					+ "6. Numero de neuronios de camada escondida (MLP) \n\n"
					+ "7. Tipo de treinamento da MLP: \n"
					+ "  -> Inserir: 'batelada' para treinamento em batelada ou 'padrao' para treinamento padrao a padrao. \n\n"
					+ "8. Numero de neuronios para a camada classe (LVQ) \n\n"
					+ "9. Inicializacao de pesos:\n"
					+ "  -> Inserir 0 para se utilizar uma matriz de pesos 0 ou "
					+ "inserir qualquer outro numero para criar uma matriz de pesos aleatorios "
					+ "com valores variando -x e x, sendo x o valor passado via este parametro. \n\n"
					+ "10. Limiar de erro. Ao atingir esse valor a rede ira encerrar seu treinamento. \n\n"
					+ "11 . Numero maximo de epocas. Ao atingir esse valor a rede ira encerrar seu treinamento.\n");
			System.exit(0);
		}
		
		if(args.length==5){			
			nome_arquivo_conjunto_dados = args[0];
			taxa_aprendizado_inicial = Double.parseDouble(args[1]);
			numero_neuronios_escondidos = Integer.parseInt(args[2]);
			numero_neuronios_classe = Integer.parseInt(args[3]);
			if(args[4].equalsIgnoreCase("z")){
				pesos_aleatorios = false;
			}else if(args[4].equalsIgnoreCase("a")){
				pesos_aleatorios = true;
			}else{
				System.out.println("Erro nos parâmetros de entrada.");
				System.exit(0);
			}
		}else if(args.length==7){
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
				System.out.println("Erro nos parâmetros de entrada.");
				System.exit(0);
			}
		}else if(args.length==11){
			
			nome_arquivo_conjunto_dados = args[0];
			
			if(args[1].equalsIgnoreCase("zeros")){
				tipo_remocao = args[1];
				valor_remocao = (int)(Integer.parseInt(args[2]));
			}else if(args[1].equalsIgnoreCase("desvio")){
				tipo_remocao = args[1];
				valor_remocao = Double.parseDouble(args[2]);
			}else if(args[1].equalsIgnoreCase("nenhum")){
				tipo_remocao = args[1];
				valor_remocao = Double.parseDouble(args[2]);
			}
			else{
				System.out.println("Erro no primeiro parametro de entrada.");
				System.exit(0);
			}
			
			if(args[3].equalsIgnoreCase("max") 
					|| args[3].equalsIgnoreCase("minmax")
					|| args[3].equalsIgnoreCase("zscore") 
					|| args[3].equalsIgnoreCase("sigmoidal")
					|| args[3].equalsIgnoreCase("nenhum")){
				tipo_normalizacao = args[3];
			}else{
				System.out.println("Erro no quarto parametro de entrada.");
				System.exit(0);
			}
			
			taxa_aprendizado_inicial = Double.parseDouble(args[4]);
			numero_neuronios_escondidos = Integer.parseInt(args[5]);
				
			if(args[6].equalsIgnoreCase("padrao")){
				modo_treinamento = 1;
			}else if(args[6].equalsIgnoreCase("batelada")){
				modo_treinamento = 2;
			}else{
				System.out.println("Erro no setimo parametro de entrada.");
				System.exit(0);
			}
			
			numero_neuronios_classe = Integer.parseInt(args[7]);
			
			if(Double.parseDouble(args[8])==0){
				pesos_aleatorios = false;
				intervalo_pesos_aleatorios = 0;
			}else if(Double.parseDouble(args[8])>=0){
				pesos_aleatorios = true;
				intervalo_pesos_aleatorios = Double.parseDouble(args[8]);
			}else{
				System.out.println("Erro no nono parametro de entrada.");
				System.exit(0);
			}
			
			limiar_erro = Double.parseDouble(args[9]);
			numero_epocas = Integer.parseInt(args[10]);
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
			
			Matrix saidas_desejadas_treinamento = situacao_problema_conjunto_dados_treinamento.get_saida();
			saidas_desejadas_treinamento = Pre_Processamento.normaliza_minmax(saidas_desejadas_treinamento);
						
			Situacao_Problema situacao_problema_conjunto_dados_validacao = Leitura_Arquivo.obtem_dados(nome_arquivo_dados_validacao);
			
			Matrix entradas_validacao = situacao_problema_conjunto_dados_validacao.get_entrada();
			
			Matrix saidas_desejadas_validacao=situacao_problema_conjunto_dados_validacao.get_saida();
			saidas_desejadas_validacao = Pre_Processamento.normaliza_minmax(saidas_desejadas_validacao);
			
			Situacao_Problema situacao_problema_conjunto_dados_teste = Leitura_Arquivo.obtem_dados(nome_arquivo_dados_teste);

			Matrix entradas_teste=situacao_problema_conjunto_dados_teste.get_entrada();
			
			Matrix saidas_desejadas_teste=situacao_problema_conjunto_dados_teste.get_saida();
			saidas_desejadas_teste = Pre_Processamento.normaliza_minmax(saidas_desejadas_teste);
			
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
		}else if(args.length==11 || args.length==5){
			Situacao_Problema situacao_problema_conjunto_dados = Leitura_Arquivo.obtem_dados(nome_arquivo_conjunto_dados);
			Matrix entradas=situacao_problema_conjunto_dados.get_entrada();
			
			if(tipo_remocao.equalsIgnoreCase("zeros")){
				entradas = Pre_Processamento.remove_zeros(entradas, (int)(valor_remocao));	
			}else if(tipo_remocao.equalsIgnoreCase("desvio")){
				entradas = Pre_Processamento.remove_desvio_baixo(entradas, valor_remocao);
			}
			
			if(tipo_normalizacao.equalsIgnoreCase("max")){
				entradas = Pre_Processamento.normaliza_max(entradas);
			}else if(tipo_normalizacao.equalsIgnoreCase("minmax")){
				entradas = Pre_Processamento.normaliza_minmax(entradas);
			}else if(tipo_normalizacao.equalsIgnoreCase("zscore")){
				entradas = Pre_Processamento.normaliza_zscore(entradas);
			}else if(tipo_normalizacao.equalsIgnoreCase("sigmoidal")){
				entradas = Pre_Processamento.normaliza_sigmoidal(entradas);
			}
			
			Matrix saidas_desejadas=situacao_problema_conjunto_dados.get_saida();
			saidas_desejadas = Pre_Processamento.normaliza_minmax(saidas_desejadas);
			
			System.out.println("#-----------Termino da Leitura Arquivo de Entrada-----#");
			
			System.out.println("\n#-----------Inicio da Separacao dos Conjuntos--------------#");
			boolean estratificado=true;
			Holdout holdout=new Holdout();
			Matrix[][] conjuntos_dados=holdout.separa_conjunto(entradas, saidas_desejadas, estratificado);
			System.out.println("#-----------Termino da Separacao dos Conjuntos------------------");
			
			classificacao_numeros = new Classificacao_Numeros(conjuntos_dados);
		}
		/*
		System.out.println("\n#----------------Inicio da Analise da MLP------------------#");
		classificacao_numeros.analisa_mlp(taxa_aprendizado_inicial, taxa_aprendizado_variavel, pesos_aleatorios, 
				numero_neuronios_escondidos, modo_treinamento, numero_epocas, intervalo_pesos_aleatorios, limiar_erro);
		System.out.println("#----------------Termino da Analise da MLP----------------#");
		*/
		System.out.println("\n#----------------Inicio da Analise da LVQ------------------#");
		classificacao_numeros.analisa_lvq(taxa_aprendizado_inicial, taxa_aprendizado_variavel, pesos_aleatorios,
				numero_neuronios_classe, numero_epocas, intervalo_pesos_aleatorios, limiar_erro);
		System.out.println("#----------------Termino da Analise da LVQ----------------#");
	}
}
