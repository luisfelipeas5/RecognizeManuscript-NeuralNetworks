import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;

import Jama.Matrix;


public class Grava_resultados {
	
	public static void grava_arquivo(String nome_arquivo, Matrix resposta_rede, String[] medidas_avaliacao, int epoca_parada){
		String novaLinha = System.getProperty("line.separator");
		
		try (Writer writer = new BufferedWriter(new OutputStreamWriter(
	              new FileOutputStream(nome_arquivo), "utf-8"))) {

			writer.write("Resultados da rede: "+novaLinha+novaLinha);
			// Imprime por linha
			for (int linha = 0; linha < resposta_rede.getRowDimension(); linha++) {
				for (int coluna = 0; coluna < resposta_rede.getColumnDimension(); coluna++) {
					writer.write(resposta_rede.get(linha, coluna)+" ");
				}
				writer.write(""+novaLinha+novaLinha);
			}
			
			writer.write(""+novaLinha+novaLinha);
			
			writer.write("Medidas de avaliacao: "+novaLinha);
			for (int i = 0; i < medidas_avaliacao.length; i++) {
				writer.write(medidas_avaliacao[i]);
			}
			
			writer.write(""+novaLinha+novaLinha);
			
			writer.write("Epoca de parada: "+epoca_parada);
			
			writer.write(""+novaLinha+novaLinha);
			
			
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public static void main(String[] args) {
		// Testando a classe:
		
		String teste1String= "teste1.txt";
		
		double[][] teste1double = new double [50][50];
		for (int i = 0; i < teste1double.length; i++) {
			for (int j = 0; j < teste1double[0].length; j++) {
				teste1double[j][i] = i*j;
			}
		}
		
		Matrix teste1Matriz = new Matrix(teste1double);
		
		String[] teste1Medidas = new String[10];
		for (int i = 0; i < teste1Medidas.length; i++) {
			teste1Medidas[i] = ("Teste "+i+"\n");
		}
		
		int epoca = 69;
		grava_arquivo(teste1String, teste1Matriz, teste1Medidas, epoca);

	}

}
