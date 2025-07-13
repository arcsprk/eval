import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import json
import base64
import io

# 테이블 스타일 상수 정의
TABLE_ROW_HEIGHT = 50  # 행 높이 (px)
TABLE_CELL_PADDING = '8px'  # 셀 패딩
TABLE_MARGIN_BOTTOM = '20px'  # 테이블 하단 마진
TABLE_MAX_HEIGHT = '400px'  # 테이블 최대 높이

# 차트 및 섹션 표시 여부 설정 (True: 표시, False: 숨김)
SHOW_RESULT_COMPARISON= True  # 좌측 시각화 차트 선택 섹션 표시 여부
SHOW_CHARTS = False  # 우측 차트 표시 여부
SHOW_VISUALIZATION_METRIC = False  # "Visualization of Metric" 섹션 표시 여부

# 샘플 데이터 생성
def generate_sample_data():
    np.random.seed(42)
    models = ['GPT-4', 'Claude-3', 'Gemini-Pro']
    prompt_templates = ['template_A', 'template_B', 'template_C']
    options_1 = ['low', 'medium', 'high']
    options_2 = ['fast', 'balanced', 'quality']
    
    data = []
    for i in range(100):
        data.append({
            'test_case_id': i,
            'model': np.random.choice(models),
            'prompt_template_name': np.random.choice(prompt_templates),
            'option-1': np.random.choice(options_1),
            'option-2': np.random.choice(options_2),
            'temperature': np.random.uniform(0.1, 1.0),
            'max_tokens': np.random.choice([1000, 2000, 4000]),
            'answer': f'Sample answer {i} - This is a longer text to test the table display and wrapping behavior.',
            'think': f'Sample thinking process {i} - This includes reasoning steps and thought process.',
            'response_time': np.random.uniform(0.5, 5.0),
            'completion_tokens': np.random.randint(100, 1000),
            'audio_url': f'https://example.com/audio_{i}.mp3' if i % 3 == 0 else '',
            'image_url': f'https://example.com/image_{i}.jpg' if i % 4 == 0 else '',
            'video_url': f'https://example.com/video_{i}.mp4' if i % 5 == 0 else '',
            'human_label': ''
        })
    
    return pd.DataFrame(data)

# 앱 초기화
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# 데이터 로드
df = generate_sample_data()

# 독립변수와 종속변수 정의
independent_vars = ['model', 'prompt_template_name', 'option-1', 'option-2', 'temperature', 'max_tokens']
metric_vars = ['response_time', 'completion_tokens']
result_vars = ['answer', 'think', 'audio_url', 'image_url', 'video_url']
all_columns = df.columns.tolist()

# 차트 타입 옵션
chart_options = [
    {'label': 'Box Plot', 'value': 'box'},
    {'label': 'Bar Chart', 'value': 'bar'},
    {'label': 'Scatter Plot', 'value': 'scatter'},
    {'label': 'Line Chart', 'value': 'line'},
    {'label': 'Histogram', 'value': 'histogram'}
]

# 공통 테이블 스타일 함수
def get_table_style():
    return {
        'style_cell': {
            'textAlign': 'left',
            'padding': TABLE_CELL_PADDING,
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '60px',
            'maxWidth': '250px',
            'width': '150px',
            'whiteSpace': 'normal',
            'height': f'{TABLE_ROW_HEIGHT}px'
        },
        'style_data_conditional': [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        'style_header': {
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'padding': TABLE_CELL_PADDING,
            'border': '1px solid #ddd',
            'height': f'{TABLE_ROW_HEIGHT}px'
        },
        'style_table': {
            'height': TABLE_MAX_HEIGHT,
            'overflowY': 'auto',
            'overflowX': 'auto',
            'minWidth': '100%',
            'border': '1px solid #ddd',
            'marginBottom': TABLE_MARGIN_BOTTOM
        },
        'style_data': {
            'border': '1px solid #ddd',
            'backgroundColor': 'white',
            'height': f'{TABLE_ROW_HEIGHT}px'
        }
    }

# 레이아웃 정의
app.layout = html.Div([
    html.H1("Prompt Test Result Analysis Dashboard", 
            style={'text-align': 'center', 'margin-bottom': '20px'}),
    
    html.Div([
        # 좌측 설정 구역
        html.Div([
            html.H3("Result Comparison Settings", style={'margin-bottom': '20px'}),
            
            # 비교할 설정 값 선택
            html.Div([
                html.H4("Comparison Settings", style={'margin-bottom': '10px'}),
                
                html.Label("Condition to Compare:"),
                dcc.Dropdown(
                    id='target-var-dropdown',
                    options=[{'label': var, 'value': var} for var in independent_vars],
                    value='model',
                    style={'margin-bottom': '15px'}
                ),
                
                html.Label("Target Values:"),
                dcc.Dropdown(
                    id='target-values-dropdown',
                    multi=True,
                    style={'margin-bottom': '20px'}
                ),
            ], style={'margin-bottom': '25px', 'padding': '15px', 
                     'border': '1px solid #ddd', 'border-radius': '5px'}),
            
            # 통제 변인 설정
            html.Div([
                html.H4("Same Settings", style={'margin-bottom': '10px'}),
                html.Div(id='control-vars-container'),
            ], style={'margin-bottom': '15px', 'padding': '10px', 
                     'border': '1px solid #ddd', 'border-radius': '5px'}),

            # 생성 결과 비교 대상 선택 (조건부 표시)
            html.Div([
                html.H4("Generation Result to Compare", style={'margin-bottom': '10px'}),
                                
                html.Label("Variables to Compare:"),
                dcc.Dropdown(
                    id='result-vars-dropdown',
                    options=[{'label': var, 'value': var} for var in result_vars],
                    value=result_vars,
                    multi=True,
                    style={'margin-bottom': '15px'}
                ),

            ], style={'margin-bottom': '15px', 'padding': '10px', 
                     'border': '1px solid #ddd', 'border-radius': '5px',
                     'display': 'block' if SHOW_RESULT_COMPARISON else 'none'}),
            
            # 차트 타입 선택 (조건부 표시)
            html.Div([
                html.H4("Visualization of Metric", style={'margin-bottom': '10px'}),
                                
                html.Label("Variables to Compare:"),
                dcc.Dropdown(
                    id='dependent-var-dropdown',
                    options=[{'label': var, 'value': var} for var in metric_vars],
                    value='response_time',
                    style={'margin-bottom': '15px'}
                ),
                dcc.Checklist(
                    id='chart-type-checklist',
                    options=chart_options,
                    value=['box'],
                    style={'margin-bottom': '15px'}
                ),

            ], style={'margin-bottom': '25px', 'padding': '15px', 
                     'border': '1px solid #ddd', 'border-radius': '5px',
                     'display': 'block' if SHOW_VISUALIZATION_METRIC else 'none'}),
            
        ], style={'width': '22%', 'float': 'left', 'padding': '15px',
                 'height': '100vh', 'overflow-y': 'auto', 'background': 'white'}),
        
        # 우측 데이터 확인 구역
        html.Div([
            # 테이블 컬럼 선택
            html.Div([
                html.Label("Columns to show:"),
                dcc.Dropdown(
                    id='table-columns-dropdown',
                    options=[{'label': col, 'value': col} for col in all_columns],
                    value=result_vars,
                    multi=True,
                    style={'margin-bottom': '15px'}
                ),
            ]),
            
            # 필터링된 데이터 테이블들 - 좌우 분할 비교
            html.Div(id='comparison-tables-container'),
            
            # 차트 섹션 (조건부 표시)
            html.Div(id='charts-container', 
                    style={'display': 'block' if SHOW_CHARTS and SHOW_VISUALIZATION_METRIC else 'none'}),
            
        ], style={'width': '78%', 'float': 'right', 'padding': '15px',
                 'height': '100vh', 'overflow-y': 'auto'}),
        
    ], style={'height': '100vh', 'display': 'flex'}),
    
    # 미디어 모달
    html.Div(id='media-modal', style={'display': 'none'}),
    
    # 데이터 저장을 위한 숨겨진 div
    html.Div(id='hidden-div', style={'display': 'none'}),
    
    # 통제 변인 값들을 저장하는 숨겨진 store
    dcc.Store(id='control-values-store')
])

# 조작 변인 선택에 따른 값 옵션 업데이트
@app.callback(
    Output('target-values-dropdown', 'options'),
    Output('target-values-dropdown', 'value'),
    Input('target-var-dropdown', 'value')
)
def update_target_values(target_var):
    if not target_var:
        return [], []
    
    unique_values = df[target_var].unique()
    options = [{'label': str(val), 'value': val} for val in unique_values]
    return options, list(unique_values)

# 통제 변인 컨트롤 생성 콜백
@app.callback(
    Output('control-vars-container', 'children'),
    Input('target-var-dropdown', 'value')
)
def update_control_vars(target_var):
    if not target_var:
        return []
    
    control_vars = [var for var in independent_vars if var != target_var]
    controls = []
    
    for var in control_vars:
        unique_values = df[var].unique()
        
        if df[var].dtype == 'object':
            options = [{'label': 'Any', 'value': 'any'}] + \
                     [{'label': str(val), 'value': val} for val in unique_values]
            control = dcc.Dropdown(
                id={'type': 'control-dropdown', 'index': var},
                options=options,
                value='any',
                style={'margin-bottom': '10px'}
            )
        else:
            # 숫자형 변수의 경우 범위 선택 또는 any
            options = [{'label': 'Any', 'value': 'any'}] + \
                     [{'label': str(val), 'value': val} for val in unique_values]
            control = dcc.Dropdown(
                id={'type': 'control-dropdown', 'index': var},
                options=options,
                value='any',
                style={'margin-bottom': '10px'}
            )
        
        controls.append(html.Div([
            html.Label(f"{var}:", style={'font-weight': 'bold'}),
            control
        ], style={'margin-bottom': '15px'}))
    
    return controls

# 통제 변인 값들을 수집하는 콜백
@app.callback(
    Output('control-values-store', 'data'),
    [Input('target-var-dropdown', 'value'),
     Input({'type': 'control-dropdown', 'index': ALL}, 'value')],
    [State({'type': 'control-dropdown', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def collect_control_values(target_var, control_values, control_ids):
    if not target_var:
        return {}
    
    control_dict = {}
    for i, control_id in enumerate(control_ids):
        var = control_id['index']
        if i < len(control_values) and control_values[i] is not None:
            control_dict[var] = control_values[i]
    
    return control_dict

# 테이블 생성 헬퍼 함수
def create_data_table(data_df, selected_columns, table_id_suffix):
    if len(data_df) == 0:
        return html.P("No data available for this condition.")
    
    # 테이블 컬럼 정의
    columns = []
    for col in selected_columns:
        if col in ['audio_url', 'image_url', 'video_url']:
            columns.append({
                'name': col.replace('_url', ''),
                'id': col,
                'presentation': 'markdown'
            })
        elif col == 'human_label':
            columns.append({
                'name': col,
                'id': col,
                'editable': True,
                'type': 'text'
            })
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric' if col in metric_vars else 'text'
            })
    
    # 미디어 URL을 버튼으로 변환
    display_df = data_df.copy()
    for media_col in ['audio_url', 'image_url', 'video_url']:
        if media_col in display_df.columns:
            display_df[media_col] = display_df[media_col].apply(
                lambda x: f"[▶️]({x})" if x else ""
            )
    
    table_styles = get_table_style()
    
    data_table = dash_table.DataTable(
        id=f'results-table-{table_id_suffix}',
        columns=columns,
        data=display_df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        page_action="native",
        page_current=0,
        page_size=10,
        **table_styles,
        fixed_rows={'headers': True},
        css=[{
            'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table',
            'rule': 'table-layout: auto;'
        }]
    )
    
    return data_table

# 데이터 필터링 및 테이블 업데이트 - 좌우 분할 비교
@app.callback(
    Output('comparison-tables-container', 'children'),
    [Input('target-var-dropdown', 'value'),
     Input('target-values-dropdown', 'value'),
     Input('table-columns-dropdown', 'value'),
     Input('result-vars-dropdown', 'value'),
     Input('control-values-store', 'data')]
)
def update_comparison_tables(target_var, target_values, selected_columns, result_vars_selected, control_dict):
    if not target_var or not target_values or not selected_columns:
        return [html.P("Please complete the settings.")]
    
    if not control_dict:
        control_dict = {}
    
    # 데이터 필터링
    filtered_df = df.copy()
    
    # 통제 변인 필터링
    for var, value in control_dict.items():
        if value is not None and value != 'any':
            filtered_df = filtered_df[filtered_df[var] == value]
    
    # 조건: target_var가 존재하고 target_values가 정확히 2개일 때만 좌우 분할 비교
    if len(target_values) == 2 and result_vars_selected and SHOW_RESULT_COMPARISON:
        # result_vars_dropdown에서 선택된 변수들과 selected_columns를 결합
        columns_to_show = []
        for col in selected_columns:
            columns_to_show.append(col)
        for col in result_vars_selected:
            if col not in columns_to_show:
                columns_to_show.append(col)
        
        # 실제 데이터프레임에 존재하는 컬럼만 필터링
        columns_to_show = [col for col in columns_to_show if col in filtered_df.columns]
        
        # 첫 번째와 두 번째 값으로 분할
        left_value = target_values[0]
        right_value = target_values[1]
        
        left_filtered_df = filtered_df[filtered_df[target_var] == left_value]
        right_filtered_df = filtered_df[filtered_df[target_var] == right_value]
        
        left_display_df = left_filtered_df[columns_to_show]
        right_display_df = right_filtered_df[columns_to_show]
        
        return [
            html.H4("Side-by-Side Comparison of Test Results", style={'margin-bottom': '20px'}),
            html.Div([
                # 좌측 테이블
                html.Div([
                    html.Div([
                        html.H5(f"{target_var}: {left_value}", 
                               style={'text-align': 'center', 'color': '#1f77b4', 'margin': '0 0 5px 0'}),
                        html.P(f"# of Test Cases: {len(left_display_df)}", 
                               style={'font-weight': 'bold', 'margin': '0 0 15px 0', 'text-align': 'center'}),
                    ], style={'height': '50px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center'}),
                    html.Div([
                        create_data_table(left_display_df, columns_to_show, 'left')
                    ], style={'height': f'calc({TABLE_MAX_HEIGHT} + 100px)'})  # 테이블 높이 통일
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top',
                         'padding': '10px', 'border': '1px solid #1f77b4', 'border-radius': '5px', 
                         'margin-right': '2%', 'box-sizing': 'border-box'}),
                
                # 우측 테이블
                html.Div([
                    html.Div([
                        html.H5(f"{target_var}: {right_value}", 
                               style={'text-align': 'center', 'color': '#ff7f0e', 'margin': '0 0 5px 0'}),
                        html.P(f"# of Test Cases: {len(right_display_df)}", 
                               style={'font-weight': 'bold', 'margin': '0 0 15px 0', 'text-align': 'center'}),
                    ], style={'height': '50px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center'}),
                    html.Div([
                        create_data_table(right_display_df, columns_to_show, 'right')
                    ], style={'height': f'calc({TABLE_MAX_HEIGHT} + 100px)'})  # 테이블 높이 통일
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top',
                         'padding': '10px', 'border': '1px solid #ff7f0e', 'border-radius': '5px',
                         'box-sizing': 'border-box'})
            ], style={'width': '100%', 'display': 'block', 'margin-bottom': '30px'})
        ]
    else:
        # 조건이 맞지 않으면 단일 테이블 표시
        single_filtered_df = filtered_df[filtered_df[target_var].isin(target_values)]
        
        # 오직 selected_columns만 사용
        columns_to_show = [col for col in selected_columns if col in single_filtered_df.columns]
        single_display_df = single_filtered_df[columns_to_show]
        
        return [
            html.H4("Results of Filtered Test Cases"),
            html.P(f"# of Test Cases: {len(single_display_df)}", 
                   style={'font-weight': 'bold', 'margin-bottom': '10px'}),
            html.P("Note: Side-by-side comparison is only available when exactly 2 target values are selected and Generation Result to Compare is enabled.", 
                   style={'color': '#666', 'font-style': 'italic', 'margin-bottom': '15px'}) 
                   if len(target_values) != 2 or not SHOW_RESULT_COMPARISON else None,
            create_data_table(single_display_df, columns_to_show, 'single')
        ]

# 차트 생성 콜백
@app.callback(
    Output('charts-container', 'children'),
    [Input('target-var-dropdown', 'value'),
     Input('target-values-dropdown', 'value'),
     Input('dependent-var-dropdown', 'value'),
     Input('chart-type-checklist', 'value'),
     Input('control-values-store', 'data')]
)
def update_charts(target_var, target_values, dependent_var, chart_types, control_dict):
    if not SHOW_VISUALIZATION_METRIC:
        return []
        
    if not target_var or not target_values or not dependent_var or not chart_types:
        return []
    
    if not control_dict:
        control_dict = {}
    
    # 데이터 필터링
    filtered_df = df.copy()
    
    # 조작 변인 필터링
    filtered_df = filtered_df[filtered_df[target_var].isin(target_values)]
    
    # 통제 변인 필터링
    for var, value in control_dict.items():
        if value is not None and value != 'any':
            filtered_df = filtered_df[filtered_df[var] == value]
    
    if len(filtered_df) == 0:
        return [html.P("No data available to generate charts.")]
    
    charts = []
    
    for chart_type in chart_types:
        if chart_type == 'box':
            fig = px.box(filtered_df, x=target_var, y=dependent_var, 
                        title=f'{dependent_var} by {target_var} (Box Plot)')
        elif chart_type == 'bar':
            avg_df = filtered_df.groupby(target_var)[dependent_var].mean().reset_index()
            fig = px.bar(avg_df, x=target_var, y=dependent_var, 
                        title=f'Average {dependent_var} by {target_var}')
        elif chart_type == 'scatter':
            fig = px.scatter(filtered_df, x=target_var, y=dependent_var, 
                           title=f'{dependent_var} by {target_var} (Scatter Plot)')
        elif chart_type == 'line':
            avg_df = filtered_df.groupby(target_var)[dependent_var].mean().reset_index()
            fig = px.line(avg_df, x=target_var, y=dependent_var, 
                         title=f'Average {dependent_var} by {target_var}')
        elif chart_type == 'histogram':
            fig = px.histogram(filtered_df, x=dependent_var, color=target_var, 
                             title=f'{dependent_var} Distribution by {target_var}')
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        
        charts.append(html.Div([
            html.H4(f"{chart_type.title()} Chart", style={'margin-top': '30px'}),
            dcc.Graph(figure=fig)
        ]))
    
    return charts

# 미디어 버튼 클릭 처리 (더미 콜백)
@app.callback(
    Output('media-modal', 'children'),
    [Input('target-var-dropdown', 'value')],  # 더미 입력
    prevent_initial_call=True
)
def handle_media_click(dummy_input):
    return html.Div([
        html.Div([
            html.H3("Media Player"),
            html.Button("Close", id="close-modal", style={'float': 'right'}),
            html.Div("Media content would appear here")
        ], style={
            'background': 'white',
            'padding': '20px',
            'border-radius': '10px',
            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'max-width': '600px',
            'margin': '100px auto'
        })
    ], style={
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '100%',
        'height': '100%',
        'background': 'rgba(0, 0, 0, 0.5)',
        'z-index': '1000'
    })

# 모달 닫기
@app.callback(
    Output('media-modal', 'style'),
    [Input('target-var-dropdown', 'value')],  # 더미 입력
    prevent_initial_call=True
)
def close_modal(dummy_input):
    return {'display': 'none'}

# 휴먼 라벨링 저장 (더미 콜백)
@app.callback(
    Output('hidden-div', 'children'),
    [Input('target-var-dropdown', 'value')],  # 더미 입력
    prevent_initial_call=True
)
def save_human_labels(dummy_input):
    return ""

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True)
