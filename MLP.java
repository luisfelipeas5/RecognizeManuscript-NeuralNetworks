//Pacote Jama: 
import Jama.Matrix; 
import java.util.Iterator;
//Estruturas de armazenamento 
import java.util.List; 
import java.util.LinkedList; 
//Instrumentos matematicos: 
import java.lang.Math; 

/**
* 
*@author Antonio 
*/

/** Classe usada para implementar a rede MLP (Multilayer Perceptron).*/
public class MLP extends Rede{
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
	
	/** Retorna uma representacao textual da rede, informando o numero de neuronios, o tipo de treinamento, a taxa de
	aprendizado inicial e se esta serah atualizada ou nao.*/
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
	
	/** Funcao sigmoide (logistica) */
	public double sigmoide(double x) {
		return 1.0/(1.0+Math.exp((-1.0)*x)); 
	}
	
	/** Derivada da funcao sigmoide */
	public double sigmoide_linha (double x) { 
		return sigmoide(x)*(1.0-sigmoide(x)); 
	}
	
	/** Aplica a funcao de ativacao (sigmoide) a cada elemento da matriz de saida da segunda parte da matriz */
	public Matrix g(Matrix x) {
		Matrix x_apf=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_valor=sigmoide(x.get(i, j));
				x_apf.set(i, j, novo_valor);
			}
		}
		return x_apf; 
	}
	
	/** Aplica a derivada da funcao de ativacao a cada elemento da matriz de saida da segunda parte da matriz */
	public Matrix g_linha(Matrix x) {
		Matrix resultado=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_elemento= sigmoide_linha(x.get(i, j));
				resultado.set(i, j, novo_elemento);
			}
		}
		return resultado;
	}
	
	/** Aplica a funcao de ativacao (sigmoide) a cada elemento da matriz de saida da primeira parte da matriz */
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
	
	/** Aplica a derivada da funcao de ativacao a cada elemento da matriz de saida da segunda parte da matriz */
	public Matrix f_linha(Matrix x) {
		Matrix resultado=new Matrix(x.getRowDimension(), x.getColumnDimension());
		for (int i = 0; i < x.getRowDimension(); i++) {
			for (int j = 0; j < x.getColumnDimension(); j++) {
				double novo_elemento= sigmoide_linha(x.get(i, j));
				resultado.set(i, j, novo_elemento);
			}
		}
		return resultado;
	}
	
	/** Calcula a saida da rede para uma dada instancia (que corresponde a uma linha da matriz de entrada e da 
	matriz de saida desejada). Alem disso, recebe como parametro as matrizes de pesos iniciais. */
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
		
		for (int i = 0; i < e.getRowDimension(); i++) {
			for (int j = 0; j < e.getColumnDimension(); j++) {
				double erro=e.get(i, j);
				erro_quadrado.set(i, j, (erro*erro));
			}
		}
		erros.add(erro_quadrado);
		return Y; 
	}
	
	/** Atualiza as matrizes de pesos baseando-se no erro obtido em uma instancia especifica e
	empregando uma taxa de aprendizado dada.*/
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
	
	/** Dadas duas matrizes, esse metodo extrai os elementos de cada uma delas e os posiciona linearmente 
	em um vetor, cujo comprimento eh igual a soma da quantidade de elementos existente nas duas matrizes. */
	Matrix calcula_vetor (Matrix A, Matrix B) {
		Matrix a = new Matrix (A.getRowPackedCopy(), A.getRowPackedCopy().length);
		Matrix b = new Matrix (B.getRowPackedCopy(), B.getRowPackedCopy().length); 
		Matrix vetor = new Matrix ((a.getRowDimension()+b.getRowDimension()),1);
		vetor.setMatrix(0,(a.getRowDimension()-1),0,(vetor.getColumnDimension()-1),a);
		vetor.setMatrix(a.getRowDimension(),(vetor.getRowDimension()-1),0,(vetor.getColumnDimension()-1),b);
		return vetor; 
	}
	
	/** Retorna o erro quadratico medio, apos aplicar na rede um dos tipos de treinamento possiveis (padrao a padrao). No caso do 
	treinamento padrao a padrao, as matrizes de pesos sao atualizadas a cada instancia; no caso do treinamento em batelada, as 
	matrizes de pesos sao atualizadas a cada epoca. Em nossa implementacao, a taxa de aprendizado usada no treinamento pode ser 
	atualizada ou nao.*/
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
				atualiza_pesos (indice_instancia, erro.get(0,0), taxa_aprendizado);
				if(atualiza_alpha) {
					taxa_aprendizado = calcula_taxa_aprendizado(erros.get(indice_instancia,0), indice_instancia, 4); 
				}				
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
			if (atualiza_alpha) {
				taxa_aprendizado = calcula_taxa_aprendizado(erros.get(erros.getRowDimension()-1,0), erros.getRowDimension()-1, 4);
			}
		}
		this.alpha = taxa_aprendizado;
		return erro_quadrado_medio;
	}
	
	/** Calcula as matrizes de gradiente dJdA e dJdB correspondentes a matrizes A e B dadas, com base no erro relacionado ao processamento
	da rede para uma instancia especifica. */
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
	
	/** Obtem uma matriz que contem todas as saidas da rede em uma dada epoca. */
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
	
	/** Passa para a rede MLP as matrizes de pesos iniciais */
	void set_pesos (Matrix pesos_a, Matrix pesos_b) {
		this.pesos_a = pesos_a; 
		this.pesos_b = pesos_b; 
	}
	
	/** Transmite para a rede as matrizes que contem as entradas e as correspondentes saidas desejadas sobre as quais o
	treinamento da rede acontecera durante a epoca.*/
	void set_problema (Matrix entrada, Matrix saida_desejada) {
		this.entradas = entrada; 
		this.saidas_desejadas = saida_desejada; 
		this.erros = new LinkedList<Matrix>(); 
		this.EQM = 0.0; 
		this.saidas_todas_instancias = new LinkedList<Matrix>();  
	}
	
	/**Informa para a rede qual o modo de treinamento desejado: padrao a padrao (1) ou batelada (2)*/	
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
	
	/** No caso da taxa de aprendizado poder ser atualizada, esse metodo calcula a melhor taxa de aprendizado dentro de um intervalo
	especificado entre os limiares taxa_inferior e taxa_superior, atraves do metodo de bissecao. Tanto o erro como o indice da instancia 
	sao requisitados pelo metodo que calcula as novas matrizes de gradientes. Alem disso, o numero de iteracoes maximo desejado para o
	metodo de bissecao tambem eh passado por parametro.*/
	double calcula_taxa_aprendizado (double erro, int indice, int num_iteracoes) {
		double taxa_inferior = 0.0; 
		double taxa_superior = 1.0; 
		Matrix d = calcula_vetor(this.dJdA, this.dJdB).times(-1.0); 
		Matrix A_novo = pesos_a.minus(this.dJdA.times(taxa_superior));
		Matrix B_novo = pesos_b.minus(this.dJdB.times(taxa_superior));
		Matrix dJdA_novo = new Matrix(A_novo.getRowDimension(), A_novo.getColumnDimension());
		Matrix dJdB_novo = new Matrix(B_novo.getRowDimension(), B_novo.getColumnDimension());
		calcula_gradientes_A_B (A_novo, B_novo, dJdA_novo, dJdB_novo, erro, indice); 
		Matrix g = calcula_vetor(dJdA_novo, dJdB_novo);
		double hl = (g.transpose().times(d)).get(0,0);  
		while (hl < 0.0) {
			d = g;
			taxa_superior = taxa_superior * 2.0;
			A_novo = pesos_a.minus(this.dJdA.times(taxa_superior)); 
			B_novo = pesos_b.minus(this.dJdB.times(taxa_superior));
			dJdA_novo = new Matrix(A_novo.getRowDimension(), A_novo.getColumnDimension()); 
			dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension()); 
			calcula_gradientes_A_B(A_novo, B_novo, dJdA_novo, dJdB_novo, erro, indice); 
			g = calcula_vetor(dJdA_novo, dJdB_novo); 
			hl = (g.transpose().times(d)).get(0,0); 
		}
		if (Math.abs(hl)<=Math.pow(10,-4)) {
			return taxa_superior;
		}
		else {
			int k = 0; 
			double taxa_media = 0.0; 
			while (k < 4 && Math.abs(hl) > Math.pow(10,-4)) {
				k++; 
				taxa_media = (taxa_inferior+taxa_superior)/2.0; 
				d = g; 
				A_novo = pesos_a.minus(this.dJdA.times(taxa_media)); 
				B_novo = pesos_b.minus(this.dJdB.times(taxa_media));
				dJdA_novo = new Matrix(A_novo.getRowDimension(), A_novo.getColumnDimension()); 
				dJdB_novo = new Matrix (B_novo.getRowDimension(), B_novo.getColumnDimension()); 
				calcula_gradientes_A_B(A_novo, B_novo, dJdA_novo, dJdB_novo, erro, indice); 
				g = calcula_vetor(dJdA_novo, dJdB_novo); 
				hl = (g.transpose().times(d)).get(0,0);
				if (hl > 0) {
					taxa_superior = taxa_media; 
				}
				else if (hl < 0) {
					taxa_inferior = taxa_media; 
				}
				else {
					break; 
				}
			}
			return taxa_media; 			
		}			
	}
}
