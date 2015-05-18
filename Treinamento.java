import java.util.Random;
import Jama.Matrix;

public class Treinamento {
	
	Rede rede;
	Matrix pesos_a;
	Matrix pesos_b;
	
	public Treinamento(Rede rede) {
		this.rede=rede;
	}

	public double treina_epoca(Matrix entradas, Matrix saidas_desejadas, boolean treina_batelada) {
		
		Matrix entrada; //instância de entrada
		Matrix saida_desejada; //saida deseja da instância de entrada
		Matrix erros=new Matrix(saidas_desejadas.getRowDimension(),	1); //matrix com cada erro quadrado de cada instancia
		
		//iteracao sobre todas as instancias do conjunto de entrada
		for(int i=0; i<entradas.getRowDimension(); i++) {
			
			//set dos conjuntos entrada e saida_desejada
			entrada=entradas.getMatrix(0, 0, 0, entradas.getColumnDimension()-1);
			saida_desejada=saidas_desejadas.getMatrix(0, 0, 0, entradas.getColumnDimension()-1);
			
			//passa para rede em questao o conjunto a instancia de entrada e recebe a saida
			Matrix saida=this.rede.calcula_saida(entrada, saida_desejada, pesos_a, pesos_b);
			
			//calcula o erro a partir da saida desejada da instancia de entrada e a saida da rede
			//Caso a rede seja uma MLP, essa matriz tera um unico elemento,
			//Caso a rede seja uma LVQ, essa matriz tera um elemento para cada neuronio de saida
			Matrix erro_matrix=saida_desejada.minus(saida);		
			double erro=0; //erro quadrado da instancia
			for(int j=0; j<erro_matrix.getColumnDimension(); j++) {
				double erro_intancia=erro_matrix.get(0, j);
				erro=erro+(erro_intancia*erro_intancia);
			}
			
			erros.set(i, 0, erro);//erro quadrado da instancia eh adicionado a matriz de erros
		}
		
		double erro_epoca=0; //erro quadratico total da epoca
		for(int i=0; i<erros.getRowDimension(); i++) {
			erro_epoca=erro_epoca+erros.get(i, 0);
		}
		
		if(treina_batelada) {
			double erro_medio=erro_epoca/entradas.getRowDimension();
			//TODO
			double taxa_aprendizado=0.1;
			rede.atualiza_pesos(erro_medio, this.pesos_a, this.pesos_b, taxa_aprendizado);
		}
		return erro_epoca;
	}
	

	public void treina(Matrix entradas_treinamento, Matrix saidas_desejadas_treinamento,
			Matrix entradas_validacao, Matrix saidas_desejadas_validacao) {
		//Para a primeira epoca, os pesos devem ser gerados randomicamente
		
		Matrix pesos_a= new Matrix( rede.numero_neuronios_escondidos, entradas_treinamento.getColumnDimension() );
		Matrix pesos_b= new Matrix( saidas_desejadas_treinamento.getColumnDimension(), rede.numero_neuronios_escondidos );
		this.gera_pesos_aleatorios(pesos_a);
		this.gera_pesos_aleatorios(pesos_b);
		
		double erro_total_treinamento=1.0;
		double erro_total_validacao=-1.0;
		
		while (erro_total_treinamento!=erro_total_validacao) {
			//erro total para o conjunto de treinamento
			boolean treina_batelada= ! rede.treina_padrao_padrao;
			erro_total_treinamento= treina_epoca(entradas_treinamento, saidas_desejadas_treinamento, treina_batelada);
			/*erro total para o conjunto de validacao
				*nao eh feita nenhum tipo de atualizacao de pesos, por isso
				*a rede nao pode atualizar padrao a padrao, nem mesmo em batch 
			*/		
			treina_batelada=false;
			boolean treina_padrao_padrao_old=rede.treina_padrao_padrao; //armazena o antigo tipo de treinamento da rede antes de mudar 
			rede.treina_padrao_padrao=false;
			erro_total_validacao= treina_epoca(entradas_validacao, saidas_desejadas_validacao, treina_batelada);
			rede.treina_padrao_padrao=treina_padrao_padrao_old;
		
			/* Printar os pesos a cada iteracao */
			int casas_decimais=3;
			System.out.println("Pesos A");
			pesos_a.print(pesos_a.getColumnDimension(), casas_decimais);
			System.out.println("Pesos B");
			pesos_b.print(pesos_b.getColumnDimension(), casas_decimais);
		}
	}
	
	/*
	 * Esse metodo, dado uma matriz pesos de dimensoes quaisquer, preenche pesos com valores aleatorios
	 * entre 0.0 e 1.0 
	 */
	public void gera_pesos_aleatorios(Matrix pesos) {
		Random random=new Random();
		for(int i=0; i< pesos.getRowDimension(); i++) {
			for(int j=0; j<pesos.getColumnDimension();i++) {
				pesos.set(i, j, random.nextDouble());
			}
		}
	}
}
