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
	
	public Matrix f_linha(Matrix x) {
		Matrix resultado=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				//double novo_elemento= sigmoide_linha(x.get(i, j));
				double novo_elemento=1;
				resultado.set(i, j, novo_elemento);
			}
		}
		return resultado;
	}
	
	
	public double get_erro() {
		double taxa_aprendizado = this.alpha;
		//saidas da epoca
		Matrix saidas=new Matrix(saidas_desejadas.getRowDimension(), saidas_desejadas.getColumnDimension());
		//erros da epoca
		Matrix erros=new Matrix(saidas_desejadas.getRowDimension(), saidas_desejadas.getColumnDimension());
		
		for (int indice_instancia = 0; indice_instancia < entradas.getRowDimension(); indice_instancia++) {
			Matrix entrada=entradas.getMatrix(indice_instancia, indice_instancia, 0, entradas.getColumnDimension()-1);
			Matrix saida_desejada=saidas_desejadas.getMatrix(indice_instancia,indice_instancia, 0, saidas_desejadas.getColumnDimension()-1);
			
			Matrix saida=calcula_saida (entrada, saida_desejada, pesos_a, pesos_b);
			
			Matrix erro=saida_desejada.minus(saida);
			
			saidas.setMatrix(indice_instancia, indice_instancia, 0, saidas.getColumnDimension()-1, saida);
			erros.setMatrix(indice_instancia, indice_instancia, 0, erros.getColumnDimension()-1, erro);
			
			if(treina_padrao_padrao && super.necessidade_atualizar_pesos) {
				/*TODO
				if(atualiza_alpha) {
					taxa_aprendizado = calcula_taxa_aprendizado(pesos_a, pesos_b, erro.get(0, 0));
				}
				*/
				this.dJdB = new Matrix (pesos_b.getRowDimension(), pesos_b.getColumnDimension());  
				this.dJdA = new Matrix (pesos_a.getRowDimension(), pesos_a.getColumnDimension());  
				calcula_gradientes_A_B (pesos_a, pesos_b, this.dJdA, this.dJdB, saida.get(0,0));
				
				Matrix gradiente_B=this.dJdB.times(taxa_aprendizado);
				Matrix novos_pesos_b=pesos_b.minus(gradiente_B);
				
				Matrix gradiente_A=this.dJdA.times(taxa_aprendizado);
				Matrix novos_pesos_a=pesos_a.minus(gradiente_A);
				
				pesos_a=novos_pesos_a;
				pesos_b=novos_pesos_b;
				/*
				System.out.println("Gradiente A");
				gradiente_A.print(gradiente_A.getColumnDimension(), 3);
				System.out.println("Pesos A");
				pesos_a.print(pesos_a.getColumnDimension(),3);
				System.out.println("Novos pesos A");
				novos_pesos_a.print(novos_pesos_a.getColumnDimension(),3);
				
				System.out.println("Gradiente B");
				gradiente_B.print(gradiente_B.getColumnDimension(), 3);
				System.out.println("Pesos A");
				pesos_b.print(pesos_b.getColumnDimension(),3);
				System.out.println("Novos pesos A");
				novos_pesos_b.print(novos_pesos_b.getColumnDimension(),3);
				*/
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
			calcula_gradientes_A_B (pesos_a, pesos_b, this.dJdA, this.dJdB, erro_quadrado_medio);
			
			Matrix gradiente_B=this.dJdB.times(taxa_aprendizado);
			pesos_b.plusEquals(gradiente_B);
			
			Matrix gradiente_A=this.dJdA.times(taxa_aprendizado);
			pesos_a.plusEquals(gradiente_A);
		}
		//TODO if (this.atualiza_alpha) {
			//this.alpha = calcula_taxa_aprendizado (pesos_a, pesos_b, this.EQM); 
		//}
		return erro_quadrado_medio;
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
		
		Matrix Z_bias = new Matrix (Z.getRowDimension(),(Z.getColumnDimension()+1)); 
		Z_bias.setMatrix(0, Z.getRowDimension()-1, 0, Z.getColumnDimension()-1, Z);
		for (int i = 0; i < Z_bias.getRowDimension(); i++) {
			Z_bias.set(i, Z_bias.getColumnDimension()-1, 1);
		}
		saidas_rede[0] = Z_bias;
		
		Matrix YIN=Z_bias.times(pesos_b.transpose());
		
		//TODO Funcao de ativacao a mesma que a saida
		//Matrix Y=f(YIN);
		//Funcao linear na saida
		Matrix Y=(YIN);
		
		semi_results[1] = YIN; 
		saidas_rede[1] = Y;
		
		Matrix e = saida_desejada.minus(Y);
		
		//Elevar erros dessa instancia ao quadrado
		for (int i = 0; i < e.getRowDimension(); i++) {
			for (int j = 0; j < e.getColumnDimension(); j++) {
				double erro=e.get(i, j);
				e.set(i, j, (erro*erro)/2);
			}
		}
		erros.add(e);

		/*
		System.out.println("Entrada:");
		entrada.print(entrada.getColumnDimension(), 3);
		System.out.println("Pesos A");
		pesos_a.print(pesos_a.getColumnDimension(), 3);
		System.out.println("ZIN:");
		ZIN.print(ZIN.getColumnDimension(), 3);
		System.out.println("Z:");
		Z.print(Z.getColumnDimension(), 3);
		System.out.println("Z BIAS:");
		Z_bias.print(Z.getColumnDimension(), 3);
		System.out.println("pesos B:");
		pesos_b.print(pesos_b.getColumnDimension(), 3);
		System.out.println("YIN:");
		YIN.print(YIN.getColumnDimension(), 3);
		System.out.println("Y:");
		Y.print(Y.getColumnDimension(), 3);
		System.out.println("erro:");
		e.print(e.getColumnDimension(), 3);
		System.out.println("erros:");
		Iterator<Matrix> iterator_erros = erros.iterator();
		while(iterator_erros.hasNext()) {
			iterator_erros.next().print(1, 3);
		}
		*/
		return Y; 
	}

	
	void calcula_gradientes_A_B (Matrix A, Matrix B, Matrix dJdA, Matrix dJdB, double erro) {
		
		Matrix YIN=semi_results[1];
		Matrix f_linha_YIN = f_linha(YIN);
		//Calculo de dJdB
		for (int i = 0; i < pesos_b.getRowDimension(); i++) {
			for (int j = 0; j < pesos_b.getColumnDimension(); j++) {
				double f_linha_YIN_k=f_linha_YIN.get(0,i);
				
				Matrix Z=saidas_rede[0];
				double Zi=Z.get(i, j);
				
				double gradiente_ponto = (f_linha_YIN_k*(erro)*(Zi));
				dJdB.set(i, j, gradiente_ponto);
				/*
				System.out.println("f_linha");
				f_linha.print(f_linha.getColumnDimension(), 3);
				System.out.println("Zi="+Zi);
				System.out.println("Gradiente no ponto b("+i+","+j+")=");
				gradiente_ponto.print(gradiente_ponto.getColumnDimension(), 3);
				*/
			}
		}
				
		Matrix erro_propagado_ate_Z=new Matrix(1,numero_neuronios);
		for (int i = 0; i < erro_propagado_ate_Z.getColumnDimension(); i++) {
			double erro_propagado_intermediario=0;
			for (int k = 0; k < saidas_desejadas.getColumnDimension(); k++) {
				double b_ki=pesos_b.get(k, i);
				double f_linha_YIN_k=f_linha_YIN.get(k,0);
				erro_propagado_intermediario+=erro*f_linha_YIN_k*b_ki;
			}
			erro_propagado_ate_Z.set(0, i, erro_propagado_intermediario);
		}
		//Calculo de dJdA
		for (int i = 0; i < pesos_a.getRowDimension(); i++) {
			for (int j = 0; j < pesos_a.getColumnDimension(); j++) {
				
				Matrix ZIN=semi_results[0];
				double f_linha_ZIN_k=sigmoide(ZIN.get(0, i));
				
				double erro_propagado_intermediario=erro_propagado_ate_Z.get(0, i);
				double gradiente_ij=erro_propagado_intermediario*f_linha_ZIN_k*entrada_instancia_atual.get(0,j);
				dJdA.set(i, j, gradiente_ij);
				/*
				System.out.println("erro_propagado_intermediario="+erro_propagado_intermediario);
				System.out.println("f_linha_ZIN_k="+f_linha_ZIN_k);
				System.out.println("entrada_instancia_atual.get(0,j);"+entrada_instancia_atual.get(0,j));
				System.out.println("gradiente_ij="+gradiente_ij);
				*/
			}
		}
		/*
		System.out.println("Erro="+erro);
		System.out.println("dJdA");
		dJdA.print(dJdA.getColumnDimension(),5);
		System.out.println("dJdB");
		dJdB.print(dJdB.getColumnDimension(),5);
		*/
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
		/*
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
		*/
	}
}
