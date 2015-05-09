import java.lang.NullPointerException; 
import Jama.Matrix; 
import java.lang.Math; 

//Classe auxiliar usada apenas para teste da rede neural MLP
public class Principal {
	public static void main (String args[]) {
		try {
			MLP rede = new MLP (10, 1.0, "optdigits.tra");
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
