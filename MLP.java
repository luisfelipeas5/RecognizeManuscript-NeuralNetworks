//Pacote Jama: 
import Jama.Matrix; 
//Estruturas de armazenamento 
import java.util.List; 
import java.util.LinkedList; 
//Instrumentos matematicos: 
import java.lang.Math; 

public class MLP extends Rede{
	/*Durante o processo "propagation", primeiro multiplicamos a matriz de entrada pela primeira matriz
	de pesos. Ao resultado, aplicamos uma funcao de ativacao (no caso deste ep, eh a funcao sigmoide).
	A matriz existente antes da aplicacao da funcao foi colocada na posicao 0 do vetor semi_results
	enquanto que a matriz resultante desta aplicacao foi salva na posicao 1 do vetor saidas_rede. Alem 
	disso, a matriz correspondente a saidas_rede[0] eh multiplicada pela segunda matriz de pesos e uma 
	nova matriz eh obtida. Essa nova matriz eh colocada na posicao 1 de semi_results. Ao aplicar uma
	nova funcao de ativacao nessa matriz (que poderia ser linear, mas, em vez disso, foi utilizada uma 
	nova sigmoide), obtem-se a matriz correspondente a posicao 1 do vetor saidas_rede*/
	
	boolean treina_padrao_padrao; 
	boolean treina_batelada;
	boolean atualiza_alpha;
	double alpha_inicial;
	
	Matrix entradas; 
	Matrix pesos_a; 
	Matrix ZINs;
	Matrix Zs;
	Matrix pesos_b;
	Matrix YINs;
	Matrix Ys;
	Matrix saidas_desejadas;
	Matrix erros_instancia;
	Matrix dJdA, dJdB;
	
	Matrix[] semi_results; 
	Matrix[] saidas_rede; 
	Matrix entrada_instancia_atual;
	Matrix saida_instancia_atual; 
	double alpha = 0.0; /*Taxa de aprendizado inicial*/  
	double EQM; 
	 
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
	
	//Funcao de sigmoide (logistica)
	public double sigmoide(double x) {
		return 1.0/(1.0+Math.exp((-1.0)*x)); 
	}
	
	//Derivada da funcao sigmoide
	public double sigmoide_linha (double x) { 
		return sigmoide(x)*(1.0-sigmoide(x)); 
	}
	
	/*Metodo que aplica a funcao de ativacao a cada elemento de uma matriz */
	public Matrix g(Matrix x) {
		//return x;
		Matrix x_apf=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_valor=sigmoide(x.get(i, j));
				x_apf.set(i, j, novo_valor);
			}
		}
		return x_apf; 
	}
	
	public Matrix g_linha(Matrix x) {
		Matrix resultado=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_elemento= sigmoide_linha(x.get(i, j));
				//double novo_elemento=1;
				resultado.set(i, j, novo_elemento);
			}
		}
		return resultado;
	}
	
	/*Metodo que aplica a funcao de ativacao a cada elemento de uma matriz */
	public Matrix f(Matrix x) {
		//return x;
		Matrix x_apf=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_valor=sigmoide(x.get(i, j));
				//double novo_valor=x.get(i, j);
				x_apf.set(i, j, novo_valor);
			}
		}
		return x_apf; 
	}
	
	public Matrix f_linha(Matrix x) {
		Matrix resultado=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_elemento= sigmoide_linha(x.get(i, j));
				//double novo_elemento=1;
				resultado.set(i, j, novo_elemento);
			}
		}
		return resultado;
	}
	
	Matrix calcula_saida(Matrix entrada, Matrix saida_desejada, Matrix pesos_a, Matrix pesos_b) {
		this.entrada_instancia_atual = entrada;
		this.saida_instancia_atual = saida_desejada; 
		semi_results = new Matrix[2];
		saidas_rede = new Matrix[2];

		Matrix ZIN = entrada.times(pesos_a.transpose());
		semi_results[0] = ZIN;
		
		Matrix Z=g(ZIN);
		
		Matrix Z_bias = new Matrix (Z.getRowDimension(),(Z.getColumnDimension()+1)); 
		Z_bias.setMatrix(0, Z.getRowDimension()-1, 0, Z.getColumnDimension()-1, Z);
		for (int i = 0; i < Z_bias.getRowDimension(); i++) {
			Z_bias.set(i, Z_bias.getColumnDimension()-1, 1);
		}
		saidas_rede[0] = Z_bias;
		
		Matrix YIN=Z_bias.times(pesos_b.transpose());
		
		Matrix Y=f(YIN);
		
		semi_results[1] = YIN; 
		saidas_rede[1] = Y;
		
		
		Matrix e = saida_desejada.minus(Y);
		
		Matrix erro_quadrado=new Matrix(e.getRowDimension(), e.getColumnDimension());
		//Elevar erros dessa instancia ao quadrado
		for (int i = 0; i < e.getRowDimension(); i++) {
			for (int j = 0; j < e.getColumnDimension(); j++) {
				double erro=e.get(i, j);
				erro_quadrado.set(i, j, (erro*erro));
			}
		}
		erros.add(erro_quadrado);
		return Y; 
	}
	
	public void atualiza_pesos (int instancia, double erro, double taxa_aprendizado) {
		this.dJdB = new Matrix (pesos_b.getRowDimension(), pesos_b.getColumnDimension());  
		this.dJdA = new Matrix (pesos_a.getRowDimension(), pesos_a.getColumnDimension());
		calcula_gradientes_A_B (pesos_a, pesos_b, this.dJdA, this.dJdB, erro, instancia);
		Matrix gradiente_B=this.dJdB.times(taxa_aprendizado);
		Matrix novos_pesos_b=pesos_b.minus(gradiente_B);
		Matrix gradiente_A=this.dJdA.times(taxa_aprendizado);
		Matrix novos_pesos_a=pesos_a.minus(gradiente_A);
		pesos_a=novos_pesos_a;
		pesos_b=novos_pesos_b;
	}
	
	Matrix calcula_vetor (Matrix A, Matrix B) {
		Matrix a = new Matrix (A.getRowPackedCopy(), A.getRowPackedCopy().length);
		Matrix b = new Matrix (B.getRowPackedCopy(), B.getRowPackedCopy().length); 
		Matrix vetor = new Matrix ((a.getRowDimension()+b.getRowDimension()),1);
		vetor.setMatrix(0,(a.getRowDimension()-1),0,(vetor.getColumnDimension()-1),a);
		vetor.setMatrix(a.getRowDimension(),(vetor.getRowDimension()-1),0,(vetor.getColumnDimension()-1),b);
		return vetor; 
	}
	
	/**/
	public double get_erro() {
		double taxa_aprendizado = this.alpha;
		//saidas da epoca
		Matrix saidas=new Matrix(saidas_desejadas.getRowDimension(), saidas_desejadas.getColumnDimension());
		//erros da epoca
		Matrix erros=new Matrix(saidas_desejadas.getRowDimension(), saidas_desejadas.getColumnDimension());
		
		ZINs=new Matrix(entradas.getRowDimension(), pesos_a.getRowDimension());
		Zs=new Matrix(entradas.getRowDimension(), pesos_a.getRowDimension()+1);
		YINs=new Matrix(entradas.getRowDimension(), pesos_b.getRowDimension());
		Ys=new Matrix(entradas.getRowDimension(), pesos_b.getRowDimension());
		
		for (int indice_instancia = 0; indice_instancia < entradas.getRowDimension(); indice_instancia++) {
			//System.out.println("------------------------------------INSTACIA="+indice_instancia);
			
			Matrix entrada=entradas.getMatrix(indice_instancia, indice_instancia, 0, entradas.getColumnDimension()-1);
			Matrix saida_desejada=saidas_desejadas.getMatrix(indice_instancia,indice_instancia, 0, saidas_desejadas.getColumnDimension()-1);
			
			Matrix saida=calcula_saida (entrada, saida_desejada, pesos_a, pesos_b);
			
			ZINs.setMatrix(indice_instancia, indice_instancia, 0, ZINs.getColumnDimension()-1, semi_results[0]);
			Zs.setMatrix(indice_instancia, indice_instancia, 0, Zs.getColumnDimension()-1, saidas_rede[0]);
			YINs.setMatrix(indice_instancia, indice_instancia, 0, YINs.getColumnDimension()-1, semi_results[1]);
			Ys.setMatrix(indice_instancia, indice_instancia, 0, Ys.getColumnDimension()-1, saida);
			
			Matrix erro=saida.minus(saida_desejada);
			
			saidas.setMatrix(indice_instancia, indice_instancia, 0, saidas.getColumnDimension()-1, saida);
			erros.setMatrix(indice_instancia, indice_instancia, 0, erros.getColumnDimension()-1, erro);
			
			if(treina_padrao_padrao && super.necessidade_atualizar_pesos) {
				/*TODO
				if(atualiza_alpha) {
					taxa_aprendizado = calcula_taxa_aprendizado(pesos_a, pesos_b, erro.get(0, 0));
				}
				*/
				atualiza_pesos ( indice_instancia, erro.get(0,0), taxa_aprendizado);
			}
		}
		//Soma dos erros
		double erro_total_quadratico=0;
		for (int indice_erro = 0; indice_erro < erros.getRowDimension(); indice_erro++) {
			double erro_quadrado=(erros.get(indice_erro,0)*erros.get(indice_erro,0))/2;
			erro_total_quadratico+=erro_quadrado;
		}
		//erro quadrado medio= (erro total quadratico) / (numero de instancias)
		double erro_quadrado_medio=erro_total_quadratico/this.entradas.getRowDimension();
		if(treina_batelada && super.necessidade_atualizar_pesos) {
			/*TODO
			if(atualiza_alpha) {
				taxa_aprendizado = calcula_taxa_aprendizado(pesos_a, pesos_b, erro.get(0, 0));
			}
			*/
			this.dJdB = new Matrix (pesos_b.getRowDimension(), pesos_b.getColumnDimension());  
			this.dJdA = new Matrix (pesos_a.getRowDimension(), pesos_a.getColumnDimension());
			
			for (int indice_erro = 0; indice_erro < erros.getRowDimension(); indice_erro++) {
				Matrix dJdA_intancia=new Matrix(dJdA.getRowDimension(), dJdA.getColumnDimension());
				Matrix dJdB_intancia=new Matrix(dJdB.getRowDimension(), dJdB.getColumnDimension());
				double erro_instancia=erros.get(indice_erro, 0);
				calcula_gradientes_A_B (pesos_a, pesos_b, dJdA_intancia, dJdB_intancia, erro_instancia, indice_erro);
				this.dJdA = this.dJdA.plus(dJdA_intancia);
				this.dJdB = this.dJdB.plus(dJdB_intancia);
			}
			
			double N_divisor= 1.0/((double)entradas.getRowDimension());
			this.dJdA = this.dJdA.times(N_divisor);
			this.dJdB = this.dJdB.times(N_divisor);
			
			Matrix gradiente_B=this.dJdB.times(taxa_aprendizado);
			Matrix novos_pesos_b=pesos_b.minus(gradiente_B);
			
			Matrix gradiente_A=this.dJdA.times(taxa_aprendizado);
			Matrix novos_pesos_a=pesos_a.minus(gradiente_A);
			
			pesos_a=novos_pesos_a;
			pesos_b=novos_pesos_b;
		}
		//TODO if (this.atualiza_alpha) {
			//this.alpha = calcula_taxa_aprendizado (pesos_a, pesos_b, this.EQM); 
		//}
		return erro_quadrado_medio;
	}
	
	void calcula_gradientes_A_B (Matrix A, Matrix B, Matrix dJdA, Matrix dJdB, double erro, int indice_instancia) {
		Matrix f_linha_YIN = f_linha(YINs.getMatrix(indice_instancia, indice_instancia, 0, YINs.getColumnDimension()-1));
		//Calculo de dJdB
		for (int i = 0; i < pesos_b.getRowDimension(); i++) {
			double f_linha_YIN_k=f_linha_YIN.get(0,i);
			for (int j = 0; j < pesos_b.getColumnDimension(); j++) {
				double Zi=Zs.get(indice_instancia, j);
				
				double gradiente_ponto = (f_linha_YIN_k*(erro)*(Zi));
				dJdB.set(i, j, gradiente_ponto);
				
			}
		}
				
		Matrix erro_propagado_ate_Z=new Matrix(1,numero_neuronios);
		for (int i = 0; i < erro_propagado_ate_Z.getColumnDimension(); i++) {
			double erro_propagado_intermediario=0;
			for (int k = 0; k < saidas_desejadas.getColumnDimension(); k++) {
				double b_ki=pesos_b.get(k, i);
				double f_linha_YIN_k=f_linha_YIN.get(0,k);
				erro_propagado_intermediario+=erro*f_linha_YIN_k*b_ki;
			}
			erro_propagado_ate_Z.set(0, i, erro_propagado_intermediario);
		}
		//Calculo de dJdA
		for (int i = 0; i < pesos_a.getRowDimension(); i++) {
			Matrix ZIN_i=ZINs.getMatrix(indice_instancia, indice_instancia,0, ZINs.getColumnDimension()-1);
			double f_linha_ZIN_k=g_linha(ZIN_i).get(0, i);
			
			for (int j = 0; j < pesos_a.getColumnDimension(); j++) {
				
				double erro_propagado_intermediario=erro_propagado_ate_Z.get(0, i);
				double gradiente_ij=erro_propagado_intermediario*f_linha_ZIN_k*entrada_instancia_atual.get(0,j);
				dJdA.set(i, j, gradiente_ij);
				
			}
		}
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
		
	/*
	double calcula_taxa_aprendizado (Matrix A, Matrix B, Matrix dJdA, Matrix dJdB) {
		double alfa = 0.0; 
		double alfa_inferior = this.alpha; 
		double alfa_superior = 1.0; //alpha_inicial 
		double ep = Math.pow(10,-3);
		double erro = 0.0; 
		Matrix aux; 
		Matrix d = calcula_vetor (dJdA, dJdB); 
		Matrix A_novo = A.minus(dJdA.times(alfa_superior)); 
		Matrix B_novo = B.minus(dJdB.times(alfa_superior));
		Matrix saida_rede = calcula_saida(entrada_instancia_atual, saida_instancia_atual, A_novo, B_novo);
		erro = saida_rede.minus(saida_instancia_atual).get(0,0); 
		Matrix dJdA_novo = new Matrix (A_novo.getRowDimension(), A_novo.getColumnDimension());
		Matrix dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension());
		calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro); 
		Matrix g = calcula_vetor (dJdA_novo, dJdB_novo); 
		Matrix gd = g.transpose().times(d); 
		double hl = gd.get(0,0); 
		while (hl < 0) {
			alfa_superior = alfa_superior*2; 
			A_novo = A.minus(dJdA_novo.times(alfa_superior)); 
			B_novo = B.minus(dJdB_novo.times(alfa_superior));
			saida_rede = calcula_saida(entrada_instancia_atual, saida_instancia_atual, A_novo, B_novo); 
			erro = saida_rede.minus(saida_instancia_atual).get(0,0);
			dJdA_novo = new Matrix (A_novo.getRowDimension(), A_novo.getColumnDimension());
			dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension());
			calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro); 
			g = calcula_vetor (dJdA_novo, dJdB_novo); 
			gd = g.transpose().times(d); 
			hl = gd.get(0,0); 
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
				saida_rede = calcula_saida(entrada_instancia_atual, saida_instancia_atual, A_novo, B_novo);
				erro = saida_rede.minus(saida_instancia_atual).get(0,0);
				dJdA_novo = new Matrix (A_novo.getRowDimension(), A_novo.getColumnDimension());
				dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension());
				calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro); 
				g = calcula_vetor (dJdA_novo, dJdB_novo);
				gd = g.transpose().times(d); 
				hl = gd.get(0,0); 
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
	}*/
}
