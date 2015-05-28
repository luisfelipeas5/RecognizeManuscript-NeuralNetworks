
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import Jama.Matrix;

public class Grafico extends ApplicationFrame{
    String titulo;
    public Grafico(final String titulo, Matrix dados) {
    	/* Basta chamar esse construtor passando o Titulo do grafico
    	 e matriz de dado que se deseja usar para se desenhar o grafico.
		 A principio trata-se de uma matriz linha que sera utilizada da seguinte forma:
    	 Uma linha representa o conjunto de erros obtidos pela rede ao longo das epocas.
    	 Cada coluna representa a Epoca daquele Erro.
    	 Ex.: Se na Epoca 1 o Erro era igual a 100 e na Epoca 2 igual a 70, esses
    	 valores seriam acessados através de:
    	 dados.get(1, 1) e dados.get(1, 2). */	
        //TESTE
    	super(titulo);
    	this.titulo=titulo;
        final JFreeChart grafico = criaGrafico(setDados(dados));
        final ChartPanel painel = new ChartPanel(grafico);
        painel.setPreferredSize(new Dimension(800, 600));
        setContentPane(painel);
    }
    
    /*
    private CategoryDataset setDados(double[][] dados_matriz) {
        final DefaultCategoryDataset dados = new DefaultCategoryDataset();
        
        for(int i = 0; i < dados_matriz.length; i++){
        	for (int j = 0; j < dados_matriz[0].length; j++) {
				//dados.addValue(dados[i][j], ""+i+"", columnKey);
			}
        }
        
    	return dados;
    }
    */
    
    private CategoryDataset setDados(Matrix dados) {
        
    	final DefaultCategoryDataset dados_dataset = new DefaultCategoryDataset();
        
        for(int linha = 0; linha < dados.getRowDimension(); linha++){
        	String legenda;
        	if(linha==0) {
        		legenda="Treinamento";
        	}else {
        		legenda="Validacao";
        	}
        	for(int coluna = 0; coluna < dados.getColumnDimension(); coluna++){
        		dados_dataset.addValue(dados.get(linha, coluna), legenda, ""+coluna+"");
        	}
        }
        
    	return dados_dataset;
    }
    
    
    private JFreeChart criaGrafico(final CategoryDataset dados) {
        
        final JFreeChart grafico = ChartFactory.createLineChart(
            this.titulo, "Epoca", "Erro", dados, PlotOrientation.VERTICAL, true, true, false                      // urls
        );

        
        final CategoryPlot plot = (CategoryPlot) grafico.getPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setRangeGridlinePaint(Color.white);

        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        rangeAxis.setAutoRangeIncludesZero(true);
		
        final LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot.getRenderer();

        // Estiliza as linhas com tracejado
        for (int i = 0; i < dados.getRowCount(); i++) {
        	renderer.setSeriesStroke(
                    i, new BasicStroke(
                        2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                        1.0f, new float[] {10.0f, 6.0f}, 0.0f
                    )
                );	
		}
        
        return grafico;
    }
    
    public static void main(final String[] args) {
    	
    	double[][] demo_d = new double[3][200];
    	for (int i = 0; i < demo_d.length; i++) {
    		for (int j = 0; j < demo_d[0].length; j++) {
    			demo_d[i][j] = 10+Math.random();
    		}	
		}
        
        Matrix demo = new Matrix(demo_d);
        Matrix demo2 = new Matrix(5, 50);
        
        for (int i = 0; i < demo2.getRowDimension(); i++) {
			for (int j = 0; j < demo2.getColumnDimension(); j++) {
				demo2.set(i, j, j*j-5*j+100*i);
			}
		}
        
        final Grafico grafico = new Grafico("Erro x Epoca", demo2);
        grafico.pack();
        RefineryUtilities.centerFrameOnScreen(grafico);
        grafico.setVisible(true);
    }
}
