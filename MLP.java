//Pacote Jama: 
import Jama.Matrix; 
import java.util.Iterator;
//Estruturas de armazenamento 
import java.util.List; 
import java.util.LinkedList; 
//Instrumentos matematicos: 
import java.lang.Math; 
import java.lang.Double; 

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
	Matrix entradas; 
	Matrix saidas_desejadas; 
	Matrix entrada_instancia_atual;
	Matrix saida_instancia_atual; 
	Matrix pesos_a; 
	Matrix pesos_b; 
	Matrix dJdA, dJdB; 
	double alpha_inicial; 
	double alpha = 0.0; /*Taxa de aprendizado inicial*/  
	double EQM; 
	boolean atualiza_alpha; 
	List<Matrix> erros; 
	List<Matrix> saidas_todas_instancias; 
	Matrix erro_instancia_atual;
		
	public MLP(int numero_neuronios_escondidos, double alpha_inicial, boolean atualiza_alpha) {
		super(numero_neuronios_escondidos);
		this.alpha = alpha_inicial; 
		this.alpha_inicial = alpha_inicial; 
		this.atualiza_alpha = atualiza_alpha; 
	}
	
	public String toString() {
		String retorno = "Rede neural MLP com " +super.numero_neuronios +" neuronios escondidos, atualizacao ";
		if (this.treina_padrao_padrao && !this.treina_batelada) {
			retorno = retorno +"padrao a padrao, "; 
		}
		else if (!this.treina_padrao_padrao && this.treina_batelada) {
			retorno = retorno +"em batelada, ";
		}
		retorno = retorno +"e taxa de aprendizado inicial " +alpha_inicial +" (a qual "; 
		if (!atualiza_alpha) {
			retorno = retorno + "nao"; 
		}
		retorno = retorno +" serah atualizada)"; 
		return retorno; 
	}
	
	//Funcao de ativacao (logistica)
	public double sigmoide(double x) {
		return 1.0/(1.0+Math.exp((-1.0)*x)); 
	}
	
	/*Metodo que aplica a funcao de ativacao a cada elemento de uma matriz */
	public Matrix f(Matrix x) {
		Matrix x_apf=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_valor=sigmoide(x.get(i, j));
				x_apf.set(i, j, novo_valor);
			}
		}
		return x_apf; 
	}
	
	
	public double get_erro() {
		if (this.EQM == 0.0) {
			treina_uma_epoca (pesos_a, pesos_b); 
		}
		return this.EQM; 
	}
	
	public Matrix get_saidas () {
		Matrix saidas_mlp=new Matrix(saidas_desejadas.getRowDimension(), saidas_desejadas.getColumnDimension());
		for (int i = 0; i < saidas_mlp.getRowDimension(); i++) {
				Matrix entrada=entradas.getMatrix(i, i, 0, entradas.getColumnDimension()-1);
				Matrix saida_desejada=saidas_desejadas.getMatrix(i, i, 0, saidas_desejadas.getColumnDimension()-1);
				calcula_saida(entrada, saida_desejada, pesos_a, pesos_b);
				
				Matrix saida=saidas_rede[1];
				saidas_mlp.setMatrix(i, i, 0, saidas_desejadas.getColumnDimension()-1, saida);
		}
		return saidas_mlp;
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
		this.entradas = entrada; 
		this.saidas_desejadas = saida_desejada; 
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
		semi_results = new Matrix[2];
		saidas_rede = new Matrix[2];
		
		Matrix ZIN = entrada.times(pesos_a.transpose()); 
		semi_results[0] = ZIN; 
		//entrada = f(ZIN); 
		Matrix Z=f(ZIN);
		saidas_rede[0] = Z; 
		
		Matrix Z_bias = new Matrix (Z.getRowDimension(),(Z.getColumnDimension()+1)); 
		Z_bias.setMatrix(0, Z.getRowDimension()-1, 0, Z.getColumnDimension()-1, Z);
		for (int i = 0; i < Z_bias.getRowDimension(); i++) {
			Z_bias.set(i, Z_bias.getColumnDimension()-1, 1);
		}
		
		Matrix YIN=Z_bias.times(pesos_b.transpose());
		Matrix Y=f(YIN);
		semi_results[1] = YIN; 
		saidas_rede[1] = Y;
		
		Matrix e = Y.minus(saida_desejada);
		this.erro_instancia_atual = e;
		//Elevar erros dessa instancia ao quadrado
		for (int i = 0; i < e.getRowDimension(); i++) {
			for (int j = 0; j < e.getColumnDimension(); j++) {
				double erro=e.get(i, j);
				e.set(i, j, (erro*erro)/2);
			}
		}
		erros.add(e);
		return Y; 
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
	
	void calcula_gradientes_A_B (Matrix A, Matrix B, Matrix dJdA, Matrix dJdB, double erro) {
		for (int i = 0; i < dJdA.getRowDimension(); i++) {
			for (int j = 0; j < dJdA.getColumnDimension(); j++) {
				dJdA.set(i,j,calcula_gradiente (false, B, i, j, erro)); 
			}
		}
		for (int i = 0; i < dJdB.getRowDimension(); i++) {
			for (int j = 0; j < dJdB.getColumnDimension(); j++) {
				dJdB.set(i,j,calcula_gradiente (true, B, i, j, erro)); 
			}
		}
	}
	
	double calcula_hl (Matrix dJdA, Matrix dJdB, Matrix dJdA_novo, Matrix dJdB_novo) {
		double hl = 0.0; 
		for (int i = 0; i < dJdA.getRowDimension(); i++) {
			for (int j = 0; j < dJdA.getColumnDimension(); j++) {
				hl = hl + dJdA.get(i,j)*dJdA_novo.get(i,j); 
			}
		}
		for (int i = 0; i < dJdB.getRowDimension(); i++) {
			for (int j = 0; j < dJdB.getColumnDimension(); j++) {
				hl = hl + dJdB.get(i,j)*dJdB_novo.get(i,j); 
			}
		}
		return hl; 
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
		this.dJdA = new Matrix (A.getRowDimension(), A.getColumnDimension());
		this.dJdB = new Matrix (B.getRowDimension(), B.getColumnDimension());
		calcula_gradientes_A_B (A, B, dJdA, dJdB, erro);
		Matrix A_novo = A.minus(dJdA.times(alfa_superior)); 
		Matrix B_novo = B.minus(dJdB.times(alfa_superior));
		if (!this.treina_batelada) {a=calcula_saida(this.entrada_instancia_atual, this.saida_instancia_atual, A_novo, B_novo);
		erro = this.erro_instancia_atual.get(0,0); }
		Matrix dJdA_novo = new Matrix (A_novo.getRowDimension(), A_novo.getColumnDimension());
		Matrix dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension());
		calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro); 
		double hl = calcula_hl (dJdA, dJdB, dJdA_novo, dJdB_novo); 
		while (hl < 0) {
			alfa_superior = alfa_superior*2; 
			A_novo = A.minus(dJdA_novo.times(alfa_superior)); 
			B_novo = B.minus(dJdB_novo.times(alfa_superior));
			if (!this.treina_batelada) { a=calcula_saida(this.entrada_instancia_atual, this.saida_instancia_atual, A_novo, B_novo); 
			erro = this.erro_instancia_atual.get(0,0);}
			dJdA_novo = new Matrix (A_novo.getRowDimension(), A_novo.getColumnDimension());
			dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension());
			calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro); 
			hl = calcula_hl (dJdA, dJdB, dJdA_novo, dJdB_novo);
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
				dJdA_novo = new Matrix (A_novo.getRowDimension(), A_novo.getColumnDimension());
				dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension());
				calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro); 
				hl = calcula_hl (dJdA, dJdB, dJdA_novo, dJdB_novo);
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
		double taxa_aprendizado; 
		taxa_aprendizado = this.alpha; 
		if (this.treina_padrao_padrao) { //atualizacao em padrao a padrao
			System.out.println ("Atualizacao padrao a padrao"); 
			for (int n = 0; n < this.entradas.getRowDimension(); n++) {
				Matrix entrada=entradas.getMatrix(n, n, 0, entradas.getColumnDimension()-1);
				Matrix saida_desejada=saidas_desejadas.getMatrix(n,n, 0, saidas_desejadas.getColumnDimension()-1);
				
				this.entrada_instancia_atual = entrada; 
				this.saida_instancia_atual = saida_desejada; 
				
				if (n==0) {
					taxa_aprendizado = calcula_taxa_aprendizado (pesos_a, pesos_b, 0.0); 
				}else {
					Matrix erro_atual=calcula_saida (entrada, saida_desejada, pesos_a, pesos_b);
					this.saidas_todas_instancias.add(erro_atual);
					this.dJdB = new Matrix (pesos_b.getRowDimension(), pesos_b.getColumnDimension());  
					this.dJdA = new Matrix (pesos_a.getRowDimension(), pesos_a.getColumnDimension());  
					calcula_gradientes_A_B (pesos_a, pesos_b, this.dJdA, this.dJdB, erro_atual.get(0,0)); 
				}
				
				if (super.necessidade_atualizar_pesos) {
					for (int i = 0; i < pesos_b.getRowDimension(); i++) {
						for (int j = 0; j < pesos_b.getColumnDimension(); j++) {
							double novo_peso=(pesos_b.get(i,j)+taxa_aprendizado*this.dJdB.get(i,j));
							pesos_b.set(i,j,novo_peso);  									
						}
					}
					for (int i = 0; i < pesos_a.getRowDimension(); i++) {
						for (int j = 0; j < pesos_a.getColumnDimension(); j++) {
							double novo_peso=(pesos_a.get(i,j)+taxa_aprendizado*this.dJdA.get(i,j));
							pesos_a.set(i,j,novo_peso);
						}
					}
				}
			}
			//Soma dos erros por instancia
			double erro_total_quadratico=0;
			Iterator<Matrix> iterator_matriz_erros_intancia = erros.iterator();
			while(iterator_matriz_erros_intancia.hasNext()) {
				Matrix matriz_erro_instacia = iterator_matriz_erros_intancia.next(); //supondo que é uma matriz linha
				for (int j = 0; j < matriz_erro_instacia.getColumnDimension(); j++) {
					erro_total_quadratico=erro_total_quadratico+matriz_erro_instacia.get(0, j);
				}
			}
			//erro quadrado medio= (erro total quadratico) / (numero de instancias)
			this.EQM = new Double(erro_total_quadratico/this.entradas.getRowDimension());
			if (this.atualiza_alpha) {
				this.alpha = calcula_taxa_aprendizado (pesos_a, pesos_b, this.EQM); 
			}
		}
		else if (this.treina_batelada) { //atualizacao em batelada 
			System.out.println ("Atualizacao em batch"); 
			for (int n = 0; n < this.entradas.getRowDimension(); n++) {
				Matrix entrada=entradas.getMatrix(n, n, 0, entradas.getColumnDimension()-1);
				Matrix saida_desejada=saidas_desejadas.getMatrix(n,n, 0, saidas_desejadas.getColumnDimension()-1);
				
				this.entrada_instancia_atual = entrada; 
				this.saida_instancia_atual = saida_desejada; 
				
				this.saidas_todas_instancias.add(calcula_saida (entrada, saida_desejada, pesos_a, pesos_b)); 
			}
			//Soma dos erros por instancia
			double erro_total_quadratico=0;
			Iterator<Matrix> iterator_matriz_erros_intancia = erros.iterator();
			while(iterator_matriz_erros_intancia.hasNext()) {
				Matrix matriz_erro_instacia = iterator_matriz_erros_intancia.next(); //supondo que é uma matriz linha
				for (int j = 0; j < matriz_erro_instacia.getColumnDimension(); j++) {
					erro_total_quadratico=erro_total_quadratico+matriz_erro_instacia.get(0, j);
				}
			}
			//erro quadrado medio= (erro total quadratico) / (numero de instancias) 
			this.EQM = new Double(erro_total_quadratico/this.entradas.getRowDimension()); 
			if (super.necessidade_atualizar_pesos) {
				for (int i = 0; i < pesos_b.getRowDimension(); i++) {
					for (int j = 0; j < pesos_b.getColumnDimension(); j++) {
						double gradiente = calcula_gradiente(true,pesos_b,i,j,this.EQM);
						double novo_peso=(pesos_b.get(i,j)+taxa_aprendizado*gradiente);
						pesos_b.set(i,j,novo_peso);  
					}
				}
				for (int i = 0; i < pesos_a.getRowDimension(); i++) {
					for (int j = 0; j < pesos_a.getColumnDimension(); j++) {
						double gradiente = calcula_gradiente(false,pesos_b,i,j,this.EQM);
						double novo_peso=(pesos_a.get(i,j)+taxa_aprendizado*gradiente);
						pesos_a.set(i,j,novo_peso);  
					}
				}
			}
			if (this.atualiza_alpha) {
				this.alpha = calcula_taxa_aprendizado (pesos_a, pesos_b, this.EQM); 
			}
		}
	}
}
