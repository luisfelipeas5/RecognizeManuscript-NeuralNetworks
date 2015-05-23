import java.lang.NullPointerException; 
import Jama.Matrix; 
import java.lang.Math; 
import java.util.Random; 

//Classe auxiliar usada apenas para teste da rede neural MLP
public class Testa_MLP {
	/*Modo de treinamento: 
	1 - treinamento padrão a padrão 
	2 - treinamento em batelada */
	public static void main (String args[]) {
		try {
			//Matriz de entrada XOR: 
			double[][] ent = {{0.0,0.0,1.0},{0.0,1.0,1.0},{1.0,0.0,1.0},{1.0,1.0,1.0}}; 
			Matrix entrada = new Matrix (ent); 
			//Matriz de saida esperada XOR
			double[][] saida = {{0.0},{1.0},{1.0},{0.0}};
			Matrix saida_esperada = new Matrix (saida);
			long inicio = System.nanoTime(); 
			//Matrizes de pesos: 
			Random r = new Random(); 
			Matrix A = new Matrix (2,entrada.getColumnDimension()); 
			Matrix B = new Matrix (saida_esperada.getColumnDimension(),3);
			for (int i = 0; i < A.getRowDimension(); i++) {
				for (int j = 0; j < A.getColumnDimension(); j++) {
					A.set(i,j,(r.nextDouble()-0.5)); 
				}
			}
			for (int i = 0; i < B.getRowDimension(); i++) {
				for (int j = 0; j < B.getColumnDimension(); j++) {
					B.set(i,j,(r.nextDouble()-0.5)); 
				}
			}
			
			//Testes: 
			//primeiro teste -- Atualização padrão a padrão com atualização de alfa
			/*Rede rede = new MLP (2, 0.9, true);
			rede.set_modo_treinamento(1);*/
			//segundo teste -- Atualização padrão a padrão sem atualização de alfa 
			/*Rede rede = new MLP (2, 0.9, false); 	
			rede.set_modo_treinamento(1);*/
			//terceiro teste -- Atualização em batch com atualização de alfa
			/*Rede rede = new MLP (2, 0.9, true); 
			rede.set_modo_treinamento (2);*/
			//quarto lote -- Atualização em batch sem atualização de alfa
			Rede rede = new MLP (2, 0.9, false); 
			rede.set_modo_treinamento (2);  
			rede.set_problema(entrada,saida_esperada); 
			rede.set_pesos(A,B); 
			//rede.set_necessidade_atualizacao(); 
			rede.get_saidas().print(0,0); 
			long fim = System.nanoTime(); 
			long intervalo = fim - inicio; 
			System.out.println ("Apos o processo de treinamento da rede MLP com base nos dados de treinamento, passaram-se (aproximadamente) " +intervalo +" nanossegundos");
		}
		catch (NullPointerException n) {
			System.out.println (n.getMessage()); 
		}		
	}
}
