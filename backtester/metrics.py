"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

from bokeh.io import show
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import output_notebook 
from bokeh.models import Div, Span, HoverTool, ColumnDataSource 


def plot_final_diagram(stats_dataframe, initial_capital):
    """
    Function that plots a summary about stats information.
    
    Parameters:
    -----------
    - stats_dataframe: pd.DataFrame that contains performances info.
    - initial_capita: float/int with predefined initial capital value
    
    Output:
    -------
    - Interactive plot of performance.
    
    Note:
    ----
    'Bokeh' visualization package was used. 
    Please, check: https://docs.bokeh.org/en/latest/index.html
    """
    
    #define main variables from stats_dataframe
    x = stats_dataframe.index.values
    
    y1 = stats_dataframe.equity_curve
    y2 = stats_dataframe.returns * 100
    y3 = stats_dataframe.drawdown * 100
    
    portfolio_value = y1 * initial_capital 
    
    #return results in the notebook
    output_notebook()
    
    #general source of information
    source = ColumnDataSource(
        data=dict(
            x=x,
            y1=y1,
            y2=y2,
            y3=y3,
            money=portfolio_value)
    )
    
    #constructing plot 1: portfolio equity curve line plot 
    plot1 = figure(plot_height=300,
                   plot_width=800,
                   y_axis_label = 'Cumulative Returns',
                   title='Portfolio Curve',
                   x_axis_type='datetime',
                   background_fill_color = None) 
    
    plot_ = plot1.line('x', 'y1', 
                       source=source, 
                       legend= 'backtest',
                       line_color= 'green', 
                       line_width=2, 
                       line_alpha=1)
    
    plot1.add_tools(HoverTool(renderers=[plot_], 
                              tooltips=[('Cash','$@money{0.000 a}')],
                              mode='vline'))
    
    plot1.legend.location = "bottom_left"
    
    daylight_savings_start = Span(location=1,
                              dimension='width', line_color='black', 
                              line_dash='dashed', line_width=1)
    
    plot1.add_layout(daylight_savings_start)
    
    plot1.title.text_font = 'arial' 
    plot1.axis.axis_label_text_font = 'arial'
    plot1.axis.axis_label_text_font_style = None
    plot1.axis.major_label_text_font_size = '9.5pt'
    plot1.title.text_font_size = '10pt'
    
    plot1.xgrid.grid_line_color = '#E4E4E4' 
    plot1.xgrid.grid_line_alpha = 0.7
    
    plot1.ygrid.grid_line_color = '#E4E4E4' 
    plot1.ygrid.grid_line_alpha = 0.7

    #constructing plot 2: returns by each heartbeat 
    plot2 = figure(plot_height=300, 
                   plot_width=800, 
                   x_range = plot1.x_range, 
                   y_axis_label = 'Returns (%)',
                   title='Heartbeat Returns (%)',
                   x_axis_type='datetime',
                   background_fill_color = None) 
    
    plot_2 = plot2.line('x','y2',
                        source=source,
                        line_width=0,
                        line_alpha=0)
    plot2.add_tools(HoverTool(renderers=[plot_2], 
                              tooltips=[('Returns','@y2{0.000 a}%')],
                              mode='vline'))

    plot2.vbar(x=stats_dataframe[stats_dataframe.returns>0].index.values,
               top=y2[y2>0], width = 4.25, 
               line_dash = 'solid',line_alpha = 0.9,
               line_width = 4.25, color='#29399F') 
    plot2.vbar(x=stats_dataframe[stats_dataframe.returns<0].index.values, 
               top=y2[y2<0],width = 4.25,
               line_dash = 'solid',line_alpha = 0.9,
               line_width = 4.25, color='#AC2E3B')
    
    plot2.xgrid.grid_line_color = None
    
    plot2.title.text_font = 'arial'
    plot2.axis.axis_label_text_font = 'arial'
    plot2.axis.axis_label_text_font_style = None
    plot2.axis.major_label_text_font_size = '9.5pt'
    plot2.title.text_font_size = '10pt'
    
    plot2.xgrid.grid_line_color = '#E4E4E4' 
    plot2.xgrid.grid_line_alpha = 0.7
    
    plot2.ygrid.grid_line_color = '#E4E4E4'
    plot2.ygrid.grid_line_alpha = 0.7
    
    #constructing plot 3: drawdowns curve by each heartbeat 
    plot3 = figure(plot_height=300, 
                   plot_width=800, 
                   x_range = plot1.x_range, 
                   x_axis_label = 'Datetime',
                   y_axis_label = 'Drawdowns (%)',
                   title='Heartbeat Drawdowns (%)',
                   x_axis_type='datetime', 
                   background_fill_color = None) 
    
    plot3.line('x', 'y3',source=source,
              line_color = '#990000', 
              line_width=3, 
              line_alpha=0.1)
    
    plot3.patch('x', 'y3',source=source, color='#D24457')
    
    plot3.title.text_font = 'arial'
    plot3.axis.axis_label_text_font = 'arial'
    plot3.axis.axis_label_text_font_style = None
    plot3.axis.major_label_text_font_size = '9.5pt'
    plot3.title.text_font_size = '10pt'
    
    plot3.xgrid.grid_line_color = '#E4E4E4' 
    plot3.xgrid.grid_line_alpha = 0.7
        
    plot3.ygrid.grid_line_color = '#E4E4E4' 
    plot3.ygrid.grid_line_alpha = 0.7
    
    plot3.add_tools(HoverTool(tooltips=[('Drawdown','@y3{0.000 a}%')],
                              mode='vline'))
    
    #linked all the plots 
    general_chart = gridplot([[plot1],
                              [plot2],
                              [plot3]], 
                             toolbar_location='right')
    
    title = Div(text="""Backtest Result""", width=150, height=25,
                style={'font-size': '150%'})
    
    #show plots as result
    show(column(title, general_chart))