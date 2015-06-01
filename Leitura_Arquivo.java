import java.io.FileReader;
import java.util.Scanner; 
import java.util.NoSuchElementException; 
import java.util.LinkedList; 
import java.lang.String; 
import java.lang.Double; 

/** Classe que serve apenas para ler o arquivo de texto. */
public class Leitura_Arquivo {
	/** Le o arquivo de entrada por meio de um objeto do tipo Scanner. Este metodo retorna um objeto da 
	classe Situacao_problema, que eh usada para compactar as matrizes de entrada e de saida esperada.*/
	public static Situacao_Problema obtem_dados (String arquivo) {
		double[][] matriz_dados;
		double[][] saida;
		Situacao_Problema sit = null; 
		LinkedList<String> linhas = new LinkedList<String>(); 
		int num_instancias = 0; 
		Scanner input=null;
		try {
			input = new Scanner (new FileReader(arquivo)); 
			while (true) {
				String aux = input.nextLine();
				linhas.add(aux); 
				num_instancias++; 
			}
		}
		catch (FileNotFoundException f) {
			System.out.println ("Arquivo de dados nao encontrado"); 
		}
		catch (NoSuchElementException e) {
			saida = new double[num_instancias][1]; 
			String l = (String) linhas.get(0); 
			String[] l_aux = l.split(",");
			int num_atributos = l_aux.length-1; 
			matriz_dados = new double[num_instancias][num_atributos];
			for (int i = 0; i < num_atributos; i++) {
				matriz_dados[0][i] = Double.parseDouble(l_aux[i]);  
			}
			saida[0][0]=Double.parseDouble(l_aux[num_atributos]);
			for (int i = 1; i < num_instancias; i++) {
				l = (String) linhas.get(i); 
				l_aux = l.split(",");
				num_atributos = l_aux.length - 1; 
				for (int j = 0; j < num_atributos; j++) {
					matriz_dados[i][j] = Double.parseDouble(l_aux[j]); 
				}
				saida[i][0] = Double.parseDouble(l_aux[num_atributos]); 
			}
			sit = new Situacao_Problema (matriz_dados, saida); 
		}
		input.close();
		return sit; 
	}
}
