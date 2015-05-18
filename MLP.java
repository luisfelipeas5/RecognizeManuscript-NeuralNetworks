//Pacote Jama: 
import Jama.Matrix; 
//Instrumentos matematicos: 
import java.lang.Math; 
//Excecoes: 
import java.lang.ArrayIndexOutOfBoundsException; 

abstract class MLP extends Rede{
	/*Durante o processo "propagation", primeiro multiplicamos a matriz de entrada pela primeira matriz
	de pesos. Ao resultado, aplicamos uma funcao de ativacao (no caso deste ep, eh a funcao sigmoide).
	A matriz existente antes da aplicacao da funcao foi colocada na posicao 0 do vetor semi_results
	enquanto que a matriz resultante desta aplicacao foi salva na posicao 1 do vetor saidas_rede. Alem 
	disso, a matriz correspondente a saidas_rede[0] eh multiplicada pela segunda matriz de pesos e uma 
	nova matriz eh obtida. Essa nova matriz eh colocada na posicao 1 de semi_results. Ao aplicar uma
	nova funcao de ativacao nessa matriz (que poderia ser linear, mas, em vez disso, foi utilizada uma 
	nova sigmoide), obtem-se a matriz correspondente a posicao 1 do vetor saidas_rede*/
	Matrix[] semi_results; 
	Matrix[] saidas_rede;  
	Matrix entrada_instancia_atual; 
	
	public MLP(double taxa_aprendizado, int numero_neuronios_escondidos, boolean treina_padrao_padrao, 
			Matrix saida, Matrix entrada) {
		super(taxa_aprendizado, numero_neuronios_escondidos, treina_padrao_padrao, saida, entrada);
	}
	
	//Funcao de ativacao (logistica)
	public double sigmoide(double x) {
		return 1.0/(1.0+Math.exp((-1.0)*x)); 
	}
	
	/*Metodo que aplica a funcao de ativacao a cada elemento de uma matriz */
	public Matrix f(Matrix x) {
		double[][] x_aux = x.getArrayCopy(); 
		double[][] x_apf = new double[x_aux.length][x_aux[0].length]; 
		for (int i = 0; i < x_apf.length; i++) {
			for (int j = 0; j < x_apf[0].length; j++) {
				x_apf[i][j] = sigmoide(x_aux[i][j]); 
			}
		}
		return new Matrix(x_apf); 
	}
	
	//Derivada da funcao de ativacao
	public double sigmoide_linha (double x) { 
		return sigmoide(x)*(1.0-sigmoide(x)); 
	}
	
	Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b) {
		this.entrada_instancia_atual = entrada; 
		Matrix entrada_aux = entrada;
		semi_results = new Matrix[2]; 
		saidas_rede = new Matrix[2];
		Matrix p = entrada_aux.times(pesos_a.transpose()); 
		semi_results[0] = p; 
		entrada_aux = f(p); 
		saidas_rede[0] = entrada_aux; 
		double[][] aux = entrada_aux.getArrayCopy(); 
		double[][] matriz_entrada = new double[entrada_aux.getRowDimension()][entrada_aux.getColumnDimension()+1]; 
		for (int i = 0; i < matriz_entrada.length; i++) {
			for (int j = 0; j < matriz_entrada[0].length-1; j++) {
				matriz_entrada[i][j] = aux[i][j]; 
			}
			matriz_entrada[i][matriz_entrada[0].length-1] = 1.0; 
		}
		entrada_aux = new Matrix(matriz_entrada);
		p = entrada_aux.times(pesos_b.transpose()); 
		semi_results[1] = p; 
		saidas_rede[1] = f(p); 
		return saidas_rede[1]; 		
	}
	
	/* 
	* ultima -> valor booleano que indica se a matriz de pesos a ser atualizada é a ultima ou não
	* pesos_b --> segunda matriz de pesos
	* (i, j) --> coordenadas do peso a ser atualizado
	* erro --> erro obtido na propagação 
	*/
	public double calcula_melhoria (boolean ultima, Matrix pesos_b, int i, int j, double erro) {
		if (ultima) {
			double ei_n = erro; 
			double[][] saida_rede_af = semi_results[1].getArrayCopy(); 
			double fl_vin = sigmoide_linha(saida_rede_af[0][i]); 
			double[][] aux = saidas_rede[0].getArrayCopy(); 
			double[][] pseudo_entrada = new double[saidas_rede[0].getRowDimension()][saidas_rede[0].getColumnDimension()+1]; 
			for (int m = 0; m < pseudo_entrada.length; m++) {
				for (int o = 0; o < pseudo_entrada[0].length -1; o++) {
					pseudo_entrada[m][o] = aux[m][o]; 
				}
				pseudo_entrada[m][pseudo_entrada[0].length - 1] = 1.0; 
			}
			double yj_n = pseudo_entrada[0][j]; 
			return super.taxa_aprendizado*ei_n*fl_vin*yj_n; 	
		}
		else {
			double e1_n = erro; 
			double[][] saida_rede_af = semi_results[1].getArrayCopy(); 
			double fl_v1n = sigmoide_linha(saida_rede_af[0][0]);
			double[][] seg_mat_pesos = pesos_b.getArrayCopy(); 
			double ei_n = seg_mat_pesos[0][i]*e1_n*fl_v1n; 				
			saida_rede_af = semi_results[0].getArrayCopy(); 
			double fl_vin = sigmoide_linha(saida_rede_af[0][i]); 
			double[][] aux = entrada_instancia_atual.getArrayCopy();
			double[][] ent = new double[entrada_instancia_atual.getRowDimension()][entrada_instancia_atual.getColumnDimension()+1]; 
			for (int m = 0; m < ent.length; m++) {
				for (int n = 0; n < ent[0].length -1; n++) {
					ent[m][n] = aux[m][n]; 
				}
				ent[m][ent[0].length - 1] = 1.0; 
			}
			double xj_n = ent[0][j];
			return super.taxa_aprendizado*ei_n*fl_vin*xj_n; 
		}
	}
	
	/*
	 * Esse metodo atualiza as matrizes de pesos dado um erro
	 */
	void atualiza_pesos(double erro, Matrix pesos_a, Matrix pesos_b) {
		try {
			if (super.treina_padrao_padrao) {
				double[][] mat = pesos_b.getArrayCopy();
				for (int i = 0; i < mat.length; i++) {
					for (int j = 0; j < mat[0].length; j++) {
						mat[i][j] = mat[i][j] + calcula_melhoria(true, pesos_b, i, j, erro); 
					}
				}
				pesos_a = new Matrix(mat); 
				mat = pesos_a.getArrayCopy();
				for (int i = 0; i < mat.length; i++) {
					for (int j = 0; j < mat[0].length; j++) {
						mat[i][j] = mat[i][j] + calcula_melhoria(false, pesos_b, i, j, erro); 
					}
				}
				pesos_b = new Matrix(mat); 
			}
		}
		catch (ArrayIndexOutOfBoundsException a) {
			System.out.println ("Erro ao acessar um campo inexistente de uma matriz. Por favor, verifique o arquivo MLP.java"); 
		}
	}
}
