import java.lang.NullPointerException; 

import Jama.Matrix; 

import java.util.Random; 

//Classe auxiliar usada apenas para teste da rede neural MLP
public class Testa_MLP {
	/*Modo de treinamento: 
	1 - treinamento padrão a padrão 
	2 - treinamento em batelada */
	public static void main (String args[]) {
		try {
			
			System.out.println("//--------------------------Teste da P1 2013---------------------------------"); 
			//Matriz de entrada Prova P1 2013
			double[][] ent3 = {{1,1,1}};
			Matrix entrada3 = new Matrix (ent3); 
			//Matriz de entrada Prova P1 2013
			double[][] saida3 = {{0,0}};
			Matrix saida_esperada3 = new Matrix (saida3);
			
			System.out.println("Enrada:");
			entrada3.print(entrada3.getColumnDimension(), 3);
			System.out.println("Saida Desejada");
			saida_esperada3.print(saida_esperada3.getColumnDimension(), 3);
			
			//Matrizes de pesos: 
			int numero_neuronios_escondidos3=3;
			Matrix A3 = new Matrix (new double[][] {{0.1,-0.1,-0.1}, {0.1,0.1,-0.1}, {-0.1,-0.1,0.1}}); 
			Matrix B3 = new Matrix (new double[][] {{0.1,-0.0,0.1,-0.1}, {-0.1,0.1,-0.1,0.1}});
			boolean atualiza_alpha3=false;
			double alpha_inicial3=0.5;
			MLP mlp3=new MLP(numero_neuronios_escondidos3, alpha_inicial3, atualiza_alpha3);
			mlp3.set_problema(entrada3, saida_esperada3);
			mlp3.set_modo_treinamento(2);
			mlp3.set_pesos(A3, B3);
			
			System.out.println("Saida antes do treinamento P1 2013=");
			Matrix saidas_p12 = mlp3.get_saidas();
			saidas_p12.print(saidas_p12.getColumnDimension(), 3);
			
			for (int i = 0; i < 10; i++) {
				System.out.println("Erro epoca: "+i);
				double erros = mlp3.get_erro();
				System.out.println(erros);
			}
			
			System.out.println("Saida apos do treinamento P1 2013=");
			saidas_p12 = mlp3.get_saidas();
			saidas_p12.print(saidas_p12.getColumnDimension(), 3);
			System.out.println("//--------------------------Teste da P1 2013--------------------------------------------------"); 
			//Matriz de entrada Prova P1 2013
			double[][] ent = {{1,1,1}};
			Matrix entrada = new Matrix (ent); 
			
			//Matriz de entrada Prova P1 2013
			double[][] saida = {{1.0}};
			Matrix saida_esperada = new Matrix (saida);
			
			System.out.println("Enrada:");
			entrada.print(entrada.getColumnDimension(), 3);
			System.out.println("Saida Desejada");
			saida_esperada.print(saida_esperada.getColumnDimension(), 3);
			//Matrizes de pesos: 
			//Random r = new Random(); 
			int numero_neuronios_escondidos=3;
			Matrix A = new Matrix (numero_neuronios_escondidos,entrada.getColumnDimension()); 
			Matrix B = new Matrix (saida_esperada.getColumnDimension(),numero_neuronios_escondidos+1);
			for (int i = 0; i < A.getRowDimension(); i++) {
				for (int j = 0; j < A.getColumnDimension(); j++) {
					//A.set(i,j,(r.nextDouble()-0.5)); 
					A.set(i,j,0.1);
				}
			}
			for (int i = 0; i < B.getRowDimension(); i++) {
				for (int j = 0; j < B.getColumnDimension(); j++) {
					//B.set(i,j,(r.nextDouble()-0.5)); 
					B.set(i,j,0.1);
				}
			}
			boolean atualiza_alpha=false;
			double alpha_inicial=0.1;
			MLP mlp=new MLP(numero_neuronios_escondidos, alpha_inicial, atualiza_alpha);
			mlp.set_problema(entrada, saida_esperada);
			mlp.set_modo_treinamento(1);
			mlp.set_pesos(A, B);
			
			System.out.println("Saida antes do treinamento=");
			Matrix saidas_p1 = mlp.get_saidas();
			saidas_p1.print(saidas_p1.getColumnDimension(), 3);
			
			for (int i = 0; i < 10; i++) {
				System.out.println("Erro epoca: "+i);
				double erros_p1 = mlp.get_erro();
				System.out.println(erros_p1);
			}
			System.out.println("Saida apos do treinamento P1 2013=");	
			saidas_p1 = mlp.get_saidas();
			saidas_p1.print(saidas_p1.getColumnDimension(), 3);
			
			System.out.println("//--------------------------XOR--------------------------------------------------"); 
			//Matriz de entrada Prova P1 2013
			double[][] ent2 = {{0.0,0.0,1.0},{0.0,1.0,1.0},{1.0,0.0,1.0},{1.0,1.0,1.0}};
			Matrix entrada2 = new Matrix (ent2); 
			//Matriz de entrada Prova P1 2013
			double[][] saida2 = {{0.0},{1.0},{1.0},{0.0}};
			Matrix saida_esperada2 = new Matrix (saida2);
			
			System.out.println("Enrada XOR:");
			entrada2.print(entrada2.getColumnDimension(), 2);
			System.out.println("Saida Desejada XOR");
			saida_esperada2.print(saida_esperada2.getColumnDimension(), 3);
			
			//Matrizes de pesos: 
			Random r2 = new Random(); 
			int numero_neuronios_escondidos2=2;
			Matrix A2 = new Matrix (numero_neuronios_escondidos2,entrada2.getColumnDimension()); 
			Matrix B2 = new Matrix (saida_esperada2.getColumnDimension(),numero_neuronios_escondidos2+1);
			for (int i = 0; i < A2.getRowDimension(); i++) {
				for (int j = 0; j < A2.getColumnDimension(); j++) {
					A2.set(i,j,(r2.nextDouble()-0.5));
					//A2.set(i,j,0.1);
				}
			}
			for (int i = 0; i < B2.getRowDimension(); i++) {
				for (int j = 0; j < B2.getColumnDimension(); j++) {
					B2.set(i,j,(r2.nextDouble()-0.5)); 
					//B2.set(i,j,0.1);
				}
			}
			
			//A2 = Pre_Processamento.normaliza_zscore(A2);
			//B2 = Pre_Processamento.normaliza_zscore(B2);
			
			boolean atualiza_alpha2=false;
			double alpha_inicial2=0.5;
			MLP mlp2=new MLP(numero_neuronios_escondidos2, alpha_inicial2, atualiza_alpha2);
			mlp2.set_problema(entrada2, saida_esperada2);
			mlp2.set_modo_treinamento(2);
			mlp2.set_pesos(A2, B2);
			
			System.out.println("Saida antes do treinamento=");
			Matrix saidas_p2 = mlp2.get_saidas();
			saidas_p2.print(saidas_p2.getColumnDimension(), 3);
			for (int i = 0; i < 100; i++) {
				System.out.println("Erro epoca: "+i);
				double erros_p2 = mlp2.get_erro();
				System.out.println(erros_p2);
			}
			System.out.println("Saida apos treinamento=");
			saidas_p2 = mlp2.get_saidas();
			saidas_p2.print(saidas_p2.getColumnDimension(), 3);
		}
		catch (NullPointerException n) {
			System.out.println (n.getMessage()); 
		}
		
	}
}
