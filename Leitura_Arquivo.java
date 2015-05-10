import Jama.Matrix; 
import java.io.FileNotFoundException; 
import java.lang.IndexOutOfBoundsException; 
import java.io.FileReader;
import java.util.Scanner; 
import java.util.NoSuchElementException; 
import java.util.LinkedList; 
import java.lang.String; 
import java.lang.Double; 

/*Classe auxiliar: 
Serve apenas para ler o arquivo de texto. Para isso, contem o metodo obtem_dados que le o arquivo de entrada por meio de um 
objeto do tipo Scanner. Este metodo retorna um objeto da classe Situacao_problema, que eh usada para compactar as matrizes 
de entrada e de saida esperada. */
public class Leitura_Arquivo {
	public static Situacao_problema obtem_dados (String arquivo) {
		double[][] matriz_dados;
		double[][] saida;
		Situacao_problema sit = null; 
		LinkedList<String> linhas = new LinkedList<String>(); 
		int num_instancias = 0; 
		try {
			Scanner input = new Scanner (new FileReader(arquivo)); 
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
			sit = new Situacao_problema (matriz_dados, saida); 
		}
		return sit; 
	}
}
