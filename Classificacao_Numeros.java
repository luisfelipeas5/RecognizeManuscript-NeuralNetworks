import Jama.Matrix;


public class Classificacao_Numeros {
	public static void main(String[] args) {
		String nome_arquivo_treinamento="treinamento.txt";
		String nome_arquivo_teste="teste.txt";
		
		Situacao_Problema situacao_problema_treinamento = Leitura_Arquivo.obtem_dados(nome_arquivo_treinamento);
		Situacao_Problema situacao_problema_teste = Leitura_Arquivo.obtem_dados(nome_arquivo_teste);
		
		Matrix entradas_treinamento=situacao_problema_treinamento.get_entrada();
		Matrix saidas_desejadas_treinamento=situacao_problema_treinamento.get_saida();
		Matrix entradas_teste=situacao_problema_teste.get_entrada();
		Matrix saidas_desejadas_teste=situacao_problema_teste.get_saida();
		
		System.out.println("Matriz de entradas treinamento sem bias");
		entradas_treinamento.print(entradas_treinamento.getColumnDimension(), 3);
		System.out.println("Matriz de saidas desejadas treinamento");
		saidas_desejadas_treinamento.print(saidas_desejadas_treinamento.getColumnDimension(), 3);
		
		System.out.println("Matriz de entradas teste sem bias");
		entradas_teste.print(entradas_teste.getColumnDimension(), 3);
		System.out.println("Matriz de saidas desejadas teste");
		saidas_desejadas_teste.print(saidas_desejadas_teste.getColumnDimension(), 3);
	}
}
