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
		//nome_arquivo_conjunto_dados="conjunto_dados.txt";
		nome_arquivo_conjunto_dados="optdigits.total.txt";
		double taxa_aprendizado_inicial=0.9;
		boolean taxa_aprendizado_variavel=true;
		boolean pesos_aleatorios=true;
		//Para MLP
		int numero_neuronios_escondidos=2;
		int modo_treinamento=1; //padrao a padrao=1 e batelada=2
		//Para LVQ
		int numero_neuronios_classe=3;
		//Numero maximo de epocas para analise
		int numero_epocas=10;
		
		/*
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
			String nome_arquivo_dados_treinamento = args[0];
			String nome_arquivo_dados_validacao = args[1];
			String nome_arquivo_dados_teste = args[2];
			
			taxa_aprendizado_inicial = Double.parseDouble(args[3]);
			numero_neuronios_escondidos = Integer.parseInt(args[4]);
			numero_neuronios_classe = Integer.parseInt(args[5]);
			if(args[6].equalsIgnoreCase("z")){
				pesos_aleatorios = true;
			}else if(args[6].equalsIgnoreCase("a")){
				pesos_aleatorios = false;
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
				pesos_aleatorios = true;
			}else if(args[4].equalsIgnoreCase("a")){
				pesos_aleatorios = false;
			}else{
				System.out.println("Erro ao inserir dados.");
				System.exit(0);
			}
		}
		*/
		
		System.out.println("\n#-----------Lendo Arquivo de Entrada------------------#");
		/*
		 * Le o arquivo do conjunto de dados e separa em:
		 * 	- Atributos (Entradas); e
		 *  - Atributos Classe (Saidas Desejadas).
		 * Colocando-os em matrizes
		 */
		Situacao_Problema situacao_problema_conjunto_dados = Leitura_Arquivo.obtem_dados(nome_arquivo_conjunto_dados);
		Matrix entradas=situacao_problema_conjunto_dados.get_entrada();
		Matrix saidas_desejadas=situacao_problema_conjunto_dados.get_saida();
		System.out.println("#-----------Termino da Leitura Arquivo de Entrada-----#");
		
		System.out.println("\n#-----------Inicio da Separacao dos Conjuntos--------------#");
		boolean estratificado=true;
		Holdout holdout=new Holdout();
		Matrix[][] conjuntos_dados=holdout.separa_conjunto(entradas, saidas_desejadas, estratificado);
		System.out.println("#-----------Termino da Separacao dos Conjuntos------------------");
		
		Classificacao_Numeros classificacao_numeros = new Classificacao_Numeros(conjuntos_dados);
		
		System.out.println("\n#----------------Inicio da Analise da MLP------------------#");
		classificacao_numeros.analisa_mlp(taxa_aprendizado_inicial, taxa_aprendizado_variavel, pesos_aleatorios,
										numero_neuronios_escondidos, modo_treinamento, numero_epocas);
		System.out.println("#----------------Termino da Analise da MLP----------------#");
		
		System.out.println("\n#----------------Inicio da Analise da LVQ------------------#");
		classificacao_numeros.analisa_lvq(taxa_aprendizado_inicial, taxa_aprendizado_variavel, pesos_aleatorios,
										numero_neuronios_classe, numero_epocas);
		System.out.println("#----------------Termino da Analise da LVQ----------------#");
	}
}
