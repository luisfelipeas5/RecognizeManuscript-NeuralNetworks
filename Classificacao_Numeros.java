import Jama.Matrix;


public class Classificacao_Numeros {
	public static void main(String[] args) {
		String nome_arquivo_treinamento=args[0];
		String nome_arquivo_teste=args[0];
		
		Situacao_Problema situacao_problema = Leitura_Arquivo.obtem_dados(nome_arquivo_treinamento);
		Matrix treinamento_entrada=situacao_problema.get_entrada();
		Matrix treinamento_saida=situacao_problema.get_saida();
		Matrix teste_entrada=situacao_problema.get_entrada();
		Matrix teste_saida=situacao_problema.get_saida();
	}
}
