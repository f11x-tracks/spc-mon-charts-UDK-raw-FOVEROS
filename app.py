import PyUber
import pandas as pd
import numpy as np
import dash
from dash import Dash, dcc, html, State, callback
from dash import dash_table as dt
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

SQL_DATA = '''
SELECT 
          a2.monitor_set_name AS monitor_set_name
         ,a5.value AS chart_value
         ,a5.test_name AS chart_test_name
         ,a0.operation AS spc_operation
         ,a1.entity AS entity
         ,To_Char(a1.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS entity_data_collect_date
         ,a10.centerline AS centerline
         ,a10.lo_control_lmt AS lo_control_lmt
         ,a10.up_control_lmt AS up_control_lmt
         ,CASE WHEN a10.centerline IS NULL THEN -99 WHEN a5.value BETWEEN a10.centerline - ((a10.centerline - a10.lo_control_lmt)/3) AND a10.centerline Then -1 WHEN a5.value BETWEEN a10.centerline AND a10.centerline + ((a10.up_control_lmt - a10.centerline)/3) THEN 1 WHEN a5.value BETWEEN a10.centerline - (2*((a10.centerline - a10.lo_control_lmt)/3)) AND a10.centerline THEN -2 WHEN a5.value BETWEEN a10.centerline AND a10.centerline + (2*((a10.up_control_lmt - a10.centerline)/3)) THEN 2 WHEN a5.value Between a10.lo_control_lmt AND a10.centerline - (2.*((a10.centerline - a10.lo_control_lmt)/3.)) THEN -3 WHEN a5.value Between a10.centerline + (2*((a10.up_control_lmt - a10.centerline)/3)) AND a10.up_control_lmt THEN 3 WHEN a5.value > a10.up_control_lmt THEN 4 WHEN a5.value < a10.lo_control_lmt THEN -4 ELSE 999 END AS zone
         ,a5.spc_chart_category AS spc_chart_category
         ,a5.spc_chart_subset AS spc_chart_subset
         ,a0.lot AS lot
         ,To_Char(a0.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS lot_data_collect_date
         ,a0.route AS route
         ,a3.parameter_class AS parameter_class
         ,a3.measurement_set_name AS measurement_set_name
         ,a2.violation_flag AS violation_flag
         ,a5.valid_flag AS chart_pt_valid_flag
         ,a5.standard_flag AS chart_standard_flag
         ,a5.chart_type AS chart_type
         ,a4.foup_slot AS foup_slot
         ,a4.wafer AS raw_wafer
         ,a4.value AS raw_value
         ,a4.wafer3 AS raw_wafer3
FROM 
P_SPC_ENTITY a1
LEFT JOIN P_SPC_Lot a0 ON a0.spcs_id = a1.spcs_id
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a1.spcs_id AND a2.data_collection_time=a1.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
INNER JOIN P_SPC_CHART_POINT a5 ON a5.spcs_id = a3.spcs_id AND a5.measurement_set_name = a3.measurement_set_name
LEFT JOIN P_SPC_CHARTPOINT_MEASUREMENT a7 ON a7.spcs_id = a3.spcs_id and a7.measurement_set_name = a3.measurement_set_name
AND a5.spcs_id = a7.spcs_id AND a5.chart_id = a7.chart_id AND a5.chart_point_seq = a7.chart_point_seq AND a5.measurement_set_name = a7.measurement_set_name
LEFT JOIN P_SPC_CHART_LIMIT a10 ON a10.chart_id = a5.chart_id AND a10.limit_id = a5.limit_id
LEFT JOIN P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
AND a4.spcs_id = a7.spcs_id AND a4.measurement_id = a7.measurement_id
WHERE
 (a2.monitor_set_name Like '%DSA_PST_NONPAT.5051.MON' or a2.monitor_set_name Like '%DSA_PST.5051.MON' or a2.monitor_set_name Like 'BARC.COATED_SURFSCAN.MFG.MON%')
 AND      a0.operation In ('8281','8333') 
 AND      (a1.entity Like 'TBC61%') 
 AND      a1.data_collection_time >= SYSDATE - 365
'''


try:
    conn = PyUber.connect(datasource='F21_PROD_XEUS')
    df = pd.read_sql(SQL_DATA, conn)
except:
    print('Cannot run SQL script - Consider connecting to VPN')

# Extract the RESIST value and create a new column
df['RESIST'] = df['SPC_CHART_CATEGORY'].str.extract(r'RESIST=([^;]+)')
# Remove the 'PARTICLE_SIZE=' portion from the 'SPC_CHART_SUBSET' column values
df['SPC_CHART_SUBSET'] = df['SPC_CHART_SUBSET'].str.replace('PARTICLE_SIZE=', '')

# Rename columns
df.rename(columns={'VIOLATION_FLAG': 'FAIL', 'CHART_PT_VALID_FLAG': 'VALID_FLAG', 'CHART_STANDARD_FLAG': 'STD_FLAG'}, inplace=True)
# Create the VALID column based on VALID_FLAG and STD_FLAG
df['VALID'] = df.apply(lambda row: 'N' if row['VALID_FLAG'] == 'N' or row['STD_FLAG'] == 'N' else 'Y', axis=1)

# Adjust ENTITY_DATA_COLLECT_DATE for each group of unique ENTITY, ENTITY_DATA_COLLECT_DATE, RAW_WAFER
df['ENTITY_DATA_COLLECT_DATE'] = pd.to_datetime(df['ENTITY_DATA_COLLECT_DATE'])  # Ensure the column is in datetime format
df = df.sort_values(by=['ENTITY', 'ENTITY_DATA_COLLECT_DATE', 'FOUP_SLOT'])  # Sort the DataFrame for consistent ordering

# Hover was not showing the different wafers for a specific run so Increment ENTITY_DATA_COLLECT_DATE by 1 minute for each wfr so you can hover over each separately
df['ENTITY_DATA_COLLECT_DATE'] += df.groupby(['ENTITY', 'ENTITY_DATA_COLLECT_DATE', 'SPC_CHART_SUBSET']).cumcount().apply(
    lambda x: timedelta(minutes=x)
)

# Save the DataFrame to an Excel file
df.to_excel('df_data.xlsx', index=False)
# df.to_csv('df_data.csv', index=False)  # Optionally save to CSV

# Initialize the Dash app
app = dash.Dash(__name__)

# Get unique defect sizes
defect_sizes = df['SPC_CHART_SUBSET'].unique()

# Layout with radio button for filtering valid data
app.layout = html.Div([
    dcc.RadioItems(
        id='only-valid',
        options=[
            {'label': 'Only Valid Data', 'value': 'Y'},
            {'label': 'All Data', 'value': 'N'}
        ],
        value='Y',  # Default value
        labelStyle={'display': 'inline-block'}
    ),
    dcc.RadioItems(
        id='y-axis-scale',  # New radio button for y-axis scaling
        options=[
            {'label': 'Auto Scale', 'value': 'auto'},
            {'label': 'Use Upper Limit', 'value': 'upper_limit'}
        ],
        value='auto',  # Default value
        labelStyle={'display': 'inline-block'}
    ),
    html.Div([
        html.Label('Moving Average Window Size:'),
        dcc.Dropdown(
            id='moving-average-window',
            options=[{'label': str(i), 'value': i} for i in range(1, 11)],
            value=2,  # Default value
            style={'width': '100px', 'display': 'inline-block'}
        )
    ], style={'margin': '10px 0'}),
    html.Div(id='charts-container')  # Container for the charts
])

# Callback to update charts based on radio button selection
@app.callback(
    Output('charts-container', 'children'),
    [Input('only-valid', 'value'),
     Input('y-axis-scale', 'value'),  # New input for y-axis scaling
     Input('moving-average-window', 'value')]  # New input for moving average window
)
def update_charts(only_valid, y_axis_scale, ma_window):
    charts_by_resist = {}
    for resist in df['RESIST'].unique():
        # Filter data by resist
        resist_df = df[df['RESIST'] == resist]
        if only_valid == 'Y':
            # Further filter data to include only valid entries
            resist_df = resist_df[resist_df['VALID'] == 'Y']
        charts = []
        for defect_size in resist_df['SPC_CHART_SUBSET'].unique():
            # Skip charts for ADDED_CLUSTER_AREA
            if defect_size == 'ADDED_CLUSTER_AREA':
                continue
            # Filter data by defect size
            defect_size_df = resist_df[resist_df['SPC_CHART_SUBSET'] == defect_size]
            # Sort data by ENTITY_DATA_COLLECT_DATE
            defect_size_df = defect_size_df.sort_values(by=['ENTITY_DATA_COLLECT_DATE', 'FOUP_SLOT'], ascending=[True, True])
            # Add jitter to the y-axis (RAW_VALUE)
            defect_size_df['RAW_VALUE_JITTERED'] = defect_size_df['RAW_VALUE'] + np.random.uniform(-0.2, 0.2, size=len(defect_size_df))  # Add jitter of ±0.2
            # Get the last entries for monitor set name, chart test name, and measurement set name
            last_monitor_set_name = defect_size_df['MONITOR_SET_NAME'].iloc[-1]
            last_chart_test_name = defect_size_df['CHART_TEST_NAME'].iloc[-1]
            last_measurement_set_name = defect_size_df['MEASUREMENT_SET_NAME'].iloc[-1]
            # Create the title for the chart
            title = f'{resist} - {defect_size}<br>{last_monitor_set_name} {last_chart_test_name}<br>{last_measurement_set_name}'
            # Calculate the upper limit for the y-axis
            upper_limit = 2 * defect_size_df['UP_CONTROL_LMT'].iloc[-1]
            # Get the center line value
            center_line = defect_size_df['CENTERLINE'].iloc[-1]
            # Create the line chart
            fig = px.scatter(defect_size_df, x='ENTITY_DATA_COLLECT_DATE', y='RAW_VALUE_JITTERED', title=title, color='ENTITY')
            # Add a horizontal line for the upper control limit
            fig.add_hline(y=defect_size_df['UP_CONTROL_LMT'].iloc[-1], line_dash="dash", annotation_text="Upper Spec", line_color="red")
            # Add a horizontal line for the center line if it exists
            if pd.notna(center_line) and center_line != '':
                fig.add_hline(y=center_line, line_dash="dash", annotation_text="Center Line")
            # Create hover text for each data point and add it as a column in the DataFrame
            defect_size_df['hovertext'] = defect_size_df.apply(
                lambda row: (
                    f'LOT: {row["LOT"]}<br>'
                    f'SLOT: {row["FOUP_SLOT"]}<br>'
                    f'WFR: {row["RAW_WAFER3"]}<br>'
                    f'ROUTE: {row["ROUTE"]}<br>'
                    f'FAIL: {row["FAIL"]}<br>'
                    f'VALID: {row["VALID"]}'
                ),
                axis=1
            )
            # Map VALID column to symbols
            valid_symbols = defect_size_df['VALID'].map({'Y': 'circle', 'N': 'x'})
            # Create the line chart with hovertext
            fig = px.line(
                defect_size_df,
                x='ENTITY_DATA_COLLECT_DATE',
                y='RAW_VALUE_JITTERED',
                title=title,
                color='ENTITY',  # Group by ENTITY
                hover_data={'hovertext': True},  # Include the hovertext column
            )

            # Add a horizontal line for the upper control limit
            fig.add_hline(
                y=defect_size_df['UP_CONTROL_LMT'].iloc[-1],
                line_dash="dash",
                annotation_text="Upper Spec",
                line_color="red"
            )

            # Add a horizontal line for the center line if it exists
            if pd.notna(center_line) and center_line != '':
                fig.add_hline(
                    y=center_line,
                    line_dash="dash",
                    annotation_text="Center Line"
                )

            # Add markers to the line chart
            fig.update_traces(
                mode='markers',  # Add both lines and markers
                marker=dict(symbol=valid_symbols)  # Use the VALID column for marker symbols
            )

            # Add trend lines for each entity
            # Get plotly default color sequence for matching entity colors
            colors = px.colors.qualitative.Plotly
            entities = list(defect_size_df['ENTITY'].unique())
            
            for i, entity in enumerate(entities):
                entity_data = defect_size_df[defect_size_df['ENTITY'] == entity].copy()
                if len(entity_data) >= ma_window:  # Need at least ma_window points for running average
                    # Sort by date for proper trend calculation
                    entity_data = entity_data.sort_values('ENTITY_DATA_COLLECT_DATE')
                    
                    # Calculate running average with dynamic window size
                    entity_data['running_avg'] = entity_data['RAW_VALUE_JITTERED'].rolling(window=ma_window, min_periods=1).mean()
                    
                    # Get the color for this entity (cycle through colors if more entities than colors)
                    entity_color = colors[i % len(colors)]
                    
                    # Add trend line to the figure
                    fig.add_trace(go.Scatter(
                        x=entity_data['ENTITY_DATA_COLLECT_DATE'],
                        y=entity_data['running_avg'],
                        mode='lines',
                        name=f'{entity} Trend',
                        line=dict(dash='dot', width=4, color=entity_color),
                        showlegend=True,  # Show in legend so it can be toggled
                        hoverinfo='skip'  # Don't show hover info for trend lines
                    ))

            # Update the layout with the y-axis range based on the selected scaling mode
            if y_axis_scale == 'upper_limit':
                fig.update_layout(
                    yaxis_range=[0, upper_limit],  # Use explicit upper limit
                )
            else:
                fig.update_layout(
                    yaxis_autorange=True  # Enable auto-scaling
                )

            # Update the layout with a transparent hover box
            fig.update_layout(
                hovermode="x unified",  # Ensure hovertext is unified across the x-axis
                hoverlabel=dict(
                    bgcolor="rgba(255, 255, 255, 0.33)",  # 33% Transparent background
                    font_size=12,
                    font_color="black",
                    bordercolor="rgba(0, 0, 0, 0)"  # Fully transparent border
                )
            )
            # Append the chart to the list
            charts.append(dcc.Graph(figure=fig))
        # Store the charts by resist
        charts_by_resist[resist] = charts

    # Create the layout with 4 columns per row for each resist
    layout = []
    for resist, charts in charts_by_resist.items():
        # Group charts into rows of 4 columns
        rows = [charts[i:i + 4] for i in range(0, len(charts), 4)]
        resist_section = html.Div([
            html.H3(f'Resist: {resist}'),  # Add a header for each resist
            html.Div([
                html.Div(
                    [
                        html.Div(
                            chart,
                            style={
                                'flex': '1',  # Make each chart take up equal space
                                'margin': '5px'  # Add some spacing between charts
                            }
                        ) for chart in row
                    ],
                    className='row',
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'width': '100%'  # Ensure the row takes the full width
                    }
                ) for row in rows
            ])
        ])
        layout.append(resist_section)

    return html.Div(layout)
    
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)