//Pacote Jama: 
import Jama.Matrix; 
//Estruturas de armazenamento 
import java.util.List; 
import java.util.LinkedList; 
//Instrumentos matematicos: 
import java.lang.Math; 
import java.lang.Double; 
//Excecoes: 
import java.lang.ArrayIndexOutOfBoundsException; 

public class MLP extends Rede{
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
	boolean treina_padrao_padrao; 
	boolean treina_batelada; 
	Matrix entrada_completa; 
	Matrix saida_desejada_completa; 
	Matrix entrada_instancia_atual;
	Matrix saida_instancia_atual; 
	Matrix pesos_a; 
	Matrix pesos_b; 
	Matrix dJdA, dJdB; 
	double alpha = 0.0; /*Taxa de aprendizado inicial*/  
	double EQM; 
	boolean atualiza_alpha; 
	List<Matrix> erros; 
	List<Matrix> saidas_todas_instancias; 
	Matrix erro_instancia_atual;
		
	public MLP(int numero_neuronios_escondidos, double alpha_inicial, boolean atualiza_alpha) {
		super(numero_neuronios_escondidos);
		this.alpha = alpha_inicial; 
		this.atualiza_alpha = atualiza_alpha; 
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
	
	
	public double get_erro() {
		if (this.EQM == 0.0) {
			treina_uma_epoca (pesos_a, pesos_b); 
		}
		return this.EQM; 
	}
	
	public Matrix get_saidas () {
		if (this.saidas_todas_instancias.size() == 0) {
			treina_uma_epoca (pesos_a, pesos_b); 
		}
		Matrix saidas_instancias = new Matrix (this.saidas_todas_instancias.size(), 1); 
		//System.out.println (this.saidas_todas_instancias.size() +" " +saidas_instancias.getRowDimension()); 
		for (int i = 0; i < this.saidas_todas_instancias.size(); i++) {
			saidas_instancias.set(i,0,this.saidas_todas_instancias.get(i).get(0,0));
		}
		return saidas_instancias;  
	}
	
	//Derivada da funcao de ativacao
	public double sigmoide_linha (double x) { 
		return sigmoide(x)*(1.0-sigmoide(x)); 
	}
	
	void set_pesos (Matrix pesos_a, Matrix pesos_b) {
		this.pesos_a = pesos_a; 
		this.pesos_b = pesos_b; 
	}
	
	void set_problema (Matrix entrada, Matrix saida_desejada) {
		this.entrada_completa = entrada; 
		this.saida_desejada_completa = saida_desejada; 
		this.erros = new LinkedList<Matrix>(); 
		this.EQM = 0.0; 
		this.saidas_todas_instancias = new LinkedList<Matrix>();  
	}
		
	void set_modo_treinamento(int modo_treinamento) {
		if (modo_treinamento == 1) { 
			this.treina_batelada = false; 
			this.treina_padrao_padrao = true; 
		}
		else if (modo_treinamento == 2) {
			this.treina_batelada = true; 
			this.treina_padrao_padrao = false; 
		}
	}
	
	Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b) {
		this.entrada_instancia_atual = entrada;
		this.saida_instancia_atual = saida_desejada; 
		Matrix entrada_aux = entrada;
		semi_results = new Matrix[2]; 
		saidas_rede = new Matrix[2];
		Matrix p = entrada_aux.times(pesos_a.transpose()); 
		semi_results[0] = p; 
		entrada_aux = f(p); 
		saidas_rede[0] = entrada_aux; 
		Matrix aux = entrada_aux; 
		entrada_aux = new Matrix (aux.getRowDimension(),(aux.getColumnDimension()+1)); 
		for (int i = 0; i < aux.getRowDimension(); i++) {
			for (int j = 0; j < aux.getColumnDimension()-1; j++) {
				entrada_aux.set(i,j,(aux.get(i,j))); 
			}
			entrada_aux.set (i, (aux.getColumnDimension()-1), 1.0); 
		}
		p = entrada_aux.times(pesos_b.transpose()); 
		semi_results[1] = p; 
		saidas_rede[1] = f(p); 	
		Matrix e = saida_desejada.minus(saidas_rede[1]); 
		this.erro_instancia_atual = e;
		e.set(0,0,(Math.pow(e.get(0,0),2)/2.0)); 
		erros.add(e);
		return saidas_rede[1]; 
	}
	
	/* 
	* ultima -> valor booleano que indica se a matriz de pesos a ser atualizada é a ultima ou não
	* pesos_b --> segunda matriz de pesos
	* (i, j) --> coordenadas do peso a ser atualizado
	* erro --> erro obtido na propagação 
	*/
	public double calcula_gradiente (boolean ultima, Matrix pesos_b, int i, int j, double erro) {
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
			return ei_n*fl_vin*yj_n; 	
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
			return ei_n*fl_vin*xj_n; 
		}
	}
	
	Matrix concatena_matrizes_vetor (Matrix A, Matrix B) {
		Matrix C = new Matrix ((A.getRowDimension()*A.getColumnDimension() + B.getRowDimension()*B.getColumnDimension()),1); 
		int i = 0; 
		while (i < C.getRowDimension()) {
			for (int m = 0; m < dJdA.getRowDimension(); m++) {
				for (int n = 0; n < dJdA.getColumnDimension(); n++) {
					C.set(i,0,dJdA.get(m,n)); 
					i++; 
				}
			}
			for (int m = 0; m < dJdB.getRowDimension(); m++) {
				for (int n = 0; n < dJdB.getColumnDimension(); n++) {
					C.set(i,0,dJdB.get(m,n)); 
					i++; 
				}
			}
		}
		return C;
	}
	
	Matrix calcula_gradientes_A_B (Matrix A, Matrix B, double erro, boolean a) {
		Matrix dJdA = new Matrix (A.getRowDimension(), A.getColumnDimension()); 
		for (int i = 0; i < dJdA.getRowDimension(); i++) {
			for (int j = 0; j < dJdA.getColumnDimension(); j++) {
				dJdA.set(i,j,calcula_gradiente (false, B, i, j, erro)); 
			}
		}
		Matrix dJdB = new Matrix (B.getRowDimension(), B.getColumnDimension()); 
		for (int i = 0; i < dJdB.getRowDimension(); i++) {
			for (int j = 0; j < dJdB.getColumnDimension(); j++) {
				dJdB.set(i,j,calcula_gradiente (true, B, i, j, erro)); 
			}
		}
		if (a) {
			return dJdA; 
		}
		else {
			return dJdB; 
		}
	}
	
	double calcula_taxa_aprendizado (Matrix A, Matrix B, double erro_medio) {
		double alfa = 0.0; 
		double alfa_inferior = 0.0; 
		double alfa_superior = this.alpha; //alpha_inicial 
		double ep = Math.pow(10,-3);
		double erro = 0.0; 
		Matrix a; 
		if (this.treina_batelada && erro_medio != 0.0) { erro = erro_medio; } 
		if (!this.treina_batelada && this.EQM == 0.0) { this.saidas_todas_instancias.add(calcula_saida(this.entrada_instancia_atual, this.saida_instancia_atual, A, B)); 
		erro = this.erro_instancia_atual.get(0,0); }
		this.dJdA = calcula_gradientes_A_B (A, B, erro, true);
		this.dJdB = calcula_gradientes_A_B (A, B, erro, false);
		Matrix d = concatena_matrizes_vetor (dJdA, dJdB); 		
		Matrix A_novo = A.minus(dJdA.times(alfa_superior)); 
		Matrix B_novo = B.minus(dJdB.times(alfa_superior));
		if (!this.treina_batelada) {a=calcula_saida(this.entrada_instancia_atual, this.saida_instancia_atual, A_novo, B_novo);
		erro = this.erro_instancia_atual.get(0,0); }
		Matrix dJdA_novo, dJdB_novo; 
		dJdA_novo = calcula_gradientes_A_B (A_novo, B_novo, erro, true);
		dJdB_novo = calcula_gradientes_A_B (A_novo, B_novo, erro, false);
		Matrix g = concatena_matrizes_vetor (dJdA_novo, dJdB_novo); 
		Matrix aux = g.transpose().times(d);
		double hl = aux.get(0,0); 
		while (hl < 0) {
			alfa_superior = alfa_superior*2; 
			A_novo = A.minus(dJdA_novo.times(alfa_superior)); 
			B_novo = B.minus(dJdB_novo.times(alfa_superior));
			if (!this.treina_batelada) { a=calcula_saida(this.entrada_instancia_atual, this.saida_instancia_atual, A_novo, B_novo); 
			erro = this.erro_instancia_atual.get(0,0);}
			dJdA_novo = calcula_gradientes_A_B (A_novo, B_novo, erro, true); 
			dJdB_novo = calcula_gradientes_A_B (A_novo, B_novo, erro, false); 
			g = concatena_matrizes_vetor (dJdA_novo, dJdB_novo); 
			aux = g.transpose().times(d);
			hl = aux.get(0,0);
		}
		if (hl >= 0 && hl <= Math.pow(10,-8)) {
			alfa = alfa_superior;
		}
		else {
			int num_interacoes = (int) Math.ceil(Math.log(alfa_superior/ep)); 
			int k = 0; 
			double alfa_medio = 0.0; 
			while (k < num_interacoes && Math.abs(hl) > Math.pow(10,-8)) {
				k++;
				alfa_medio = (alfa_inferior+alfa_superior)/2.0; 
				A_novo = A.minus(dJdA_novo.times(alfa_medio)); 
				B_novo = B.minus(dJdB_novo.times(alfa_medio));
				if (!this.treina_batelada) { a=calcula_saida(this.entrada_instancia_atual, this.saida_instancia_atual, A_novo, B_novo);
				erro = this.erro_instancia_atual.get(0,0);}
				dJdA_novo = calcula_gradientes_A_B (A_novo, B_novo, erro, true); 
				dJdB_novo = calcula_gradientes_A_B (A_novo, B_novo, erro, false); 
				g = concatena_matrizes_vetor (dJdA_novo, dJdB_novo); 
				aux = g.transpose().times(d);
				hl = aux.get(0,0);
				if (hl > Math.pow(10,-8)) {
					alfa_superior = alfa_medio; 
				}
				else if (hl < 0) {
					alfa_inferior = alfa_medio; 
				}
				else if (hl >= 0 && hl <= Math.pow(10,-8)){
					break; 
				}				 
			}
			alfa = alfa_medio;
		}
		return alfa; 
	}
	/*
	 * Esse metodo atualiza as matrizes de pesos dado um erro
	 */
	void treina_uma_epoca(Matrix pesos_a, Matrix pesos_b) {
		try {
			double taxa_aprendizado; 
			taxa_aprendizado = this.alpha; 
			if (this.treina_padrao_padrao && !this.treina_batelada) { //atualizacao em padrao a padrao
				//System.out.println ("Atualizacao padrao a padrao"); 
				for (int n = 0; n < this.entrada_completa.getRowDimension(); n++) {
					Matrix ent = new Matrix(1, this.entrada_completa.getColumnDimension()); 
					for (int j = 0; j < ent.getColumnDimension(); j++) {
						ent.set(0,j,this.entrada_completa.get(0,j)); 
					}
					Matrix saida = new Matrix(1, this.saida_desejada_completa.getColumnDimension()); 
					for (int j = 0; j < saida.getColumnDimension(); j++) {
						saida.set(0,j,this.saida_desejada_completa.get(0,j)); 
					}
					this.entrada_instancia_atual = ent; 
					this.saida_instancia_atual = saida; 
					if (atualiza_alpha) {
						taxa_aprendizado = calcula_taxa_aprendizado (pesos_a, pesos_b, 0.0); 
					}
					else {
						//System.out.println ("Nao atualiza alpha"); 
						this.saidas_todas_instancias.add(calcula_saida (ent, saida, pesos_a, pesos_b));						
						this.dJdB = new Matrix (pesos_b.getRowDimension(), pesos_b.getColumnDimension());
						for (int i = 0; i < this.dJdB.getRowDimension(); i++) {
							for (int j = 0; j < this.dJdB.getColumnDimension(); j++) {
								this.dJdB.set(i,j,calcula_gradiente(true,pesos_b,i,j,this.erro_instancia_atual.get(0,0))); 
							}
						}
						this.dJdA = new Matrix (pesos_a.getRowDimension(), pesos_a.getColumnDimension());
						for (int i = 0; i < this.dJdA.getRowDimension(); i++) {
							for (int j = 0; j < this.dJdA.getColumnDimension(); j++) {
								this.dJdA.set(i,j,calcula_gradiente(false,pesos_b,i,j,this.erro_instancia_atual.get(0,0))); 
							}
						}
					}
					if (super.necessidade_atualizar_pesos) {
						for (int i = 0; i < pesos_b.getRowDimension(); i++) {
							for (int j = 0; j < pesos_b.getColumnDimension(); j++) {
								if (n == 0) {
								pesos_b.set(i,j,(pesos_b.get(i,j)+this.alpha*this.dJdB.get(i,j)));  
								} else {
								pesos_b.set(i,j,(pesos_b.get(i,j)+taxa_aprendizado*this.dJdB.get(i,j)));  	
								}
							}
						}
						for (int i = 0; i < pesos_a.getRowDimension(); i++) {
							for (int j = 0; j < pesos_a.getColumnDimension(); j++) {
								if (n == 0) {
								pesos_a.set(i,j,(pesos_a.get(i,j)+this.alpha*this.dJdA.get(i,j)));  
								}
								else {
								pesos_a.set(i,j,(pesos_a.get(i,j)+taxa_aprendizado*this.dJdA.get(i,j)));
								}
							}
						}
					}
				}
				double Et = 0.0; 
				for (int pos = 0; pos < this.erros.size(); pos++) {
					Et = Et + erros.get(pos).get(0,0); 
				}
				this.EQM = new Double(Et/this.entrada_completa.getRowDimension()); 		
				if (this.atualiza_alpha) { this.alpha = calcula_taxa_aprendizado (pesos_a, pesos_b, this.EQM); }
			}
			else if (!this.treina_padrao_padrao && this.treina_batelada) { //atualizacao em batelada 
				//System.out.println ("Atualizacao em batch"); 
				Matrix ent = new Matrix(1, this.entrada_completa.getColumnDimension()); 
				Matrix saida = new Matrix(1, this.saida_desejada_completa.getColumnDimension()); 
				for (int n = 0; n < this.entrada_completa.getRowDimension(); n++) {
					for (int j = 0; j < ent.getColumnDimension(); j++) {
						ent.set(0,j,this.entrada_completa.get(n,j)); 
					}
					for (int j = 0; j < saida.getColumnDimension(); j++) {
						saida.set(0,j,this.saida_desejada_completa.get(n,j)); 
					}
					this.saidas_todas_instancias.add(calcula_saida (ent, saida, pesos_a, pesos_b)); 
				}
				double Et = 0.0; 
				for (int pos = 0; pos < this.erros.size(); pos++) {
					Et = Et + erros.get(pos).get(0,0); 
				}
				this.EQM = new Double(Et/this.entrada_completa.getRowDimension()); 
				if (super.necessidade_atualizar_pesos) {
					for (int i = 0; i < pesos_b.getRowDimension(); i++) {
						for (int j = 0; j < pesos_b.getColumnDimension(); j++) {
							pesos_b.set(i,j,(pesos_b.get(i,j)+taxa_aprendizado*calcula_gradiente(true,pesos_b,i,j,this.EQM)));  
						}
					}
					for (int i = 0; i < pesos_a.getRowDimension(); i++) {
						for (int j = 0; j < pesos_a.getColumnDimension(); j++) {
							pesos_a.set(i,j,(pesos_a.get(i,j)+taxa_aprendizado*calcula_gradiente(false,pesos_b,i,j,this.EQM)));  
						}
					}
				}
				if (this.atualiza_alpha) { this.alpha = calcula_taxa_aprendizado (pesos_a, pesos_b, this.EQM); }
			}
		}
		catch (ArrayIndexOutOfBoundsException a) {
			System.out.println ("Erro ao acessar um campo inexistente de uma matriz. Por favor, verifique o arquivo MLP.java"); 
		}
	}
}
