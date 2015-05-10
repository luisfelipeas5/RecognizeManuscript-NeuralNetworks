import java.lang.NullPointerException; 
import Jama.Matrix; 
import java.lang.Math; 

//Classe auxiliar usada apenas para teste da rede neural MLP
public class Testa_MLP {
	public static void main (String args[]) {
		try {
			String arquivo_dados=args[0];
			Situacao_problema s = Leitura_Arquivo.obtem_dados(arquivo_dados);
			MLP rede = new MLP (10, 1.0, s.get_entrada(), s.get_saida());
			long inicio = System.nanoTime(); 
			rede.inicializa_rede(); 
			rede.propagation(); 
			rede.backpropagation();
			long fim = System.nanoTime(); 
			long transicao = (long) Math.pow(10,9); 
			long intervalo = (fim - inicio)/transicao; 
			System.out.println ("Apos o processo de treinamento da rede MLP com base nos dados de treinamento, passaram-se (aproximadamente) " +intervalo +" segundos");
		}
		catch (NullPointerException n) {
			System.out.println (""); 
		}
	}
}
