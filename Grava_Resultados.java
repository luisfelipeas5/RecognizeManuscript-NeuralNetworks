import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;

import Jama.Matrix;

/**Classe que possui a funcao de gerar um arquivo com os resultados da rede*/
public class Grava_Resultados {
	/**Grava em um arquivo ".txt" de saida as respostas da rede, bem como as medidas de avalizacao e a epoca em que houve o termino 
	da execucao */
	public static void grava_arquivo(String nome_arquivo, Matrix resposta_rede, String[][] medidas_avaliacao, int epoca_parada){
		String novaLinha = System.getProperty("line.separator");
		
		try (Writer writer = new BufferedWriter(new OutputStreamWriter(
	              new FileOutputStream(nome_arquivo+"_resultados.txt"), "utf-8"))) {

			writer.write("Resultados da rede: "+novaLinha+novaLinha);
			// Imprime por linha
			for (int linha = 0; linha < resposta_rede.getRowDimension(); linha++) {
				for (int coluna = 0; coluna < resposta_rede.getColumnDimension(); coluna++) {
					writer.write(String.format("%.3f", resposta_rede.get(linha, coluna)));
				}
				writer.write(""+novaLinha);
			}
			
			writer.write(""+novaLinha+novaLinha);
			
			writer.write("Epoca de parada: "+epoca_parada);
			
			writer.write(""+novaLinha+novaLinha);
			
			
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try (Writer writer = new BufferedWriter(new OutputStreamWriter(
	              new FileOutputStream(nome_arquivo+"_medidas_avaliacao.csv"), "utf-8"))) {
			
			for (int i = 0; i < medidas_avaliacao[1].length; i++) {
				if(medidas_avaliacao[1][i]!=null)
				writer.write(medidas_avaliacao[1][i]);
			}
			
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**Transforma a matriz que contem a saida da rede para o conjunto de teste em uma matriz de String */
	public static String[][] toString(Matrix dados){
		// Matrix dados eh uma matriz linha com a saida da rede para o conjunto de teste
		String[][] resultado = new String[dados.getRowDimension()][dados.getColumnDimension()];
		for (int linha = 0; linha< dados.getRowDimension(); linha++) {
			for (int coluna = 0; coluna < dados.getColumnDimension(); coluna++) {
				resultado[linha][coluna]=(""+dados.get(linha, coluna));
			}
		}
		
		return resultado;
	}
	
	/**Metodo que fora usado para testar esta classe */
	public static void main(String[] args) {
		//String teste1String= "teste1.txt";
		
		double[][] teste1double = new double [50][50];
		for (int i = 0; i < teste1double.length; i++) {
			for (int j = 0; j < teste1double[0].length; j++) {
				teste1double[j][i] = i*j;
			}
		}
		
		//Matrix teste1Matriz = new Matrix(teste1double);
		
		String[] teste1Medidas = new String[10];
		for (int i = 0; i < teste1Medidas.length; i++) {
			teste1Medidas[i] = ("Teste "+i+"\n");
		}
		
		//int epoca = 69;
		//grava_arquivo(teste1String, teste1Matriz, teste1Medidas, epoca);

	}

}
