import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PLOT 1: RMSE ---

def chart_epoch_loss(
    dataframe ,
    x_col_name='epoch',
    y_col_names=['training_rmse', 'testing_rmse'],
    y_col_colour_map={'training_rmse': 'cyan', 'testing_rmse': 'red'},
    y_col_line_styles={'training_rmse': 'solid', 'testing_rmse': 'dash'},
    title="RMSE over Epochs",):
    
    fig = px.line(
        dataframe, 
        x=x_col_name, 
        y=y_col_names, 
        title=title,
        #labels={'value': 'RMSE', 'variable': 'Type'},
        color_discrete_map=y_col_colour_map
    )
    # Add markers to all lines
    fig.update_traces(mode='lines+markers')

    # Apply line styles from the y_col_line_styles mapping, if provided
    if y_col_line_styles:
        for trace_name, dash_style in y_col_line_styles.items():
            if dash_style is None:
                continue
            fig.update_traces(
                selector={"name": trace_name},
                line={"dash": dash_style}
            )

    # Move the legend to the top center, under the title
    fig.update_layout(
        legend=dict(
            orientation="h",  # 'h' for horizontal layout
            yanchor="bottom", # Anchor the legend's bottom edge
            y=1.02,           # Position the legend just above the plot area (1.0)
            xanchor="center", # Anchor the legend's center
            x=0.5             # Center the legend horizontally
        )
    )

    #fig.show()
    return fig



def chart_dual_QQ(
    dataframe,
    x_col_name= 'payment_size',
    y_col_name= 'pred_ffnn_claims', ):

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Test QQ Plot (Original Values)', 'Test QQ Plot (Log-transformed Values)'))

    # --- Plot 1: QQ Plot with Original Values ---
    max_val = max(dataframe[x_col_name].max(), dataframe[y_col_name].max())

    fig.add_trace(
        go.Scatter(
            x=dataframe[x_col_name],
            y=dataframe[y_col_name],
            mode='markers',
            name='Average by Quantile',
            marker=dict(color='blue', opacity=0.7),
            showlegend=False  # Disable global legend
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Ideal Prediction (y=x)',
            line=dict(color='red', dash='dash'),
            showlegend=False  # Disable global legend
        ),
        row=1, col=1
    )

    # --- Plot 2: QQ Plot with Log Values ---
    log_pred = np.log1p(dataframe[y_col_name])
    log_actual = np.log1p(dataframe[x_col_name])
    max_log_val = max(log_pred.max(), log_actual.max())

    fig.add_trace(
        go.Scatter(
            x=log_actual,
            y=log_pred,
            mode='markers',
            #name='Average by Quantile (Log)',
            marker=dict(color='blue', opacity=0.7),
            showlegend=False  # Disable global legend
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_log_val],
            y=[0, max_log_val],
            mode='lines',
            #name='Ideal Prediction (y=x) (Log)',
            line=dict(color='red', dash='dash'),
            showlegend=False  # Disable global legend
        ),
        row=1, col=2
    )

    # Add separate legend annotations in the top left of each subplot
    fig.add_annotation(
        text="<b>Legend</b><br>● Average by Quantile<br>━ Ideal Prediction (y=x)",
        #xref="x1", yref="y1",  # Reference to first subplot's axes
        # Set the reference frame for x and y to 'paper'
        xref='paper', 
        yref='paper',
        x=0, y=1,  # Top left position
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )

    fig.add_annotation(
        text="<b>Legend</b><br>● Average by Quantile (Log)<br>━ Ideal Prediction (y=x) (Log)",
        xref='paper', 
        yref='paper',
        x=0.75, y=1,  # Top left position
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )

    # Update axes
    fig.update_xaxes(title_text='Actual Average per Quantile', row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text='Predicted Average per Quantile', row=1, col=1, showgrid=True)
    fig.update_xaxes(title_text='Actual Average per Quantile (log-transformed)', row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text='Predicted Average per Quantile (log-transformed)', row=1, col=2, showgrid=True)

    fig.update_layout(height=500, width=1000, title_text="QQ Plots for Test Set")

    #fig.show()

    return fig
