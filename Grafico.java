
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

public class Grafico extends ApplicationFrame{
    
    public Grafico(final String titulo, double[] dados) {
        super(titulo);
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
    
    private CategoryDataset setDados(double[] dados_double) {
        final DefaultCategoryDataset dados_dataset = new DefaultCategoryDataset();
        
        for(int i = 0; i < dados_double.length; i++){
        	dados_dataset.addValue(dados_double[i], "Serie 1", ""+i+"");
        }
        
    	return dados_dataset;
    }
    
    
    private JFreeChart criaGrafico(final CategoryDataset dados) {
        
        final JFreeChart grafico = ChartFactory.createLineChart(
            "Erro X Epocas", "Epoca", "Erro", dados, PlotOrientation.VERTICAL, true, true, false                      // urls
        );

        
        final CategoryPlot plot = (CategoryPlot) grafico.getPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setRangeGridlinePaint(Color.white);

        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        rangeAxis.setAutoRangeIncludesZero(true);
		
        final LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot.getRenderer();
//        renderer.setDrawShapes(true);

        renderer.setSeriesStroke(
            0, new BasicStroke(
                2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                1.0f, new float[] {10.0f, 6.0f}, 0.0f
            )
        );
        
        return grafico;
    }
    
    public static void main(final String[] args) {
    	
    	double[] demo = new double[100];
        for (int i = 0; i < demo.length; i++) {
			demo[i] = Math.random();
		}
        final Grafico grafico = new Grafico("Erro X Epoca", demo);
        grafico.pack();
        RefineryUtilities.centerFrameOnScreen(grafico);
        grafico.setVisible(true);
    }
}
