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
import os

# dash_player 임포트 (비디오 재생용)
try:
    import dash_player as dp
    DASH_PLAYER_AVAILABLE = True
except ImportError:
    DASH_PLAYER_AVAILABLE = False
    print("Warning: dash_player not available. Install with: pip install dash-player")

# 테이블 스타일 상수 정의
TABLE_ROW_HEIGHT = 50  # 행 높이 (px)
TABLE_CELL_PADDING = '8px'  # 셀 패딩
TABLE_MARGIN_BOTTOM = '20px'  # 테이블 하단 마진
TABLE_MAX_HEIGHT = '400px'  # 테이블 최대 높이

# 차트 및 섹션 표시 여부 설정 (True: 표시, False: 숨김)
SHOW_CHARTS = False  # 우측 차트 표시 여부
SHOW_VISUALIZATION_METRIC = False  # "Visualization of Metric" 섹션 표시 여부

# 오디오 표시 설정
SHOW_AUDIO_PLAYER_IN_TABLE = True  # True: 테이블에 오디오 플레이어 표시, False: 아이콘/파일명만 표시
VISIBLE_AUDIO_FILE_NAME = True  # SHOW_AUDIO_PLAYER_IN_TABLE이 False일 때만 적용

# 이미지 표시 설정
SHOW_IMAGE_THUMBNAILS = True  # True: 테이블에 썸네일 표시, False: 아이콘만 표시
IMAGE_THUMBNAIL_SIZE = 40  # 테이블 내 이미지 썸네일 크기 (픽셀)
IMAGE_HOVER_PREVIEW_SIZE = 200  # 호버 시 프리뷰 크기 (픽셀)

# 테이블 복사 허용 설정 (True: 복사 허용, False: 복사 금지)
ALLOW_COPY = True

# 샘플 데이터 생성
def generate_sample_data():
    np.random.seed(42)
    models = ['GPT-4', 'Claude-3', 'Gemini-Pro']
    prompt_templates = ['template_A', 'template_B', 'template_C']
    options_1 = ['low', 'medium', 'high']
    options_2 = ['fast', 'balanced', 'quality']
    
    data = []
    # 다양한 미디어 URL들
    audio_urls = [
        'https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3',  # MP3
        'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav',  # WAV
        'https://file-examples.com/storage/fe68c065dfa8fcaa80f2b0d/2017/11/file_example_OOG_1MG.ogg',  # OGG
    ]
    
    video_urls = [
        'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
        'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
        'https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4',
    ]
    
    image_urls = [
        'https://picsum.photos/300/200?random=1',
        'https://picsum.photos/300/200?random=2',
        'https://picsum.photos/300/200?random=3',
        'https://via.placeholder.com/300x200/FF0000/FFFFFF?text=Sample+1',
        'https://via.placeholder.com/300x200/00FF00/FFFFFF?text=Sample+2',
    ]
    
    for i in range(100):
        data.append({
            'test_case_id': i,
            'model': np.random.choice(models),
            'prompt_template_name': np.random.choice(prompt_templates),
            'option-1': np.random.choice(options_1),
            'option-2': np.random.choice(options_2),
            'temperature': np.random.uniform(0.1, 1.0),
            'max_tokens': np.random.choice([1000, 2000, 4000]),
            'answer': f'## Sample answer {i} \n ### Section 2 \n - This is a longer text to test the table display and wrapping behavior.',
            'think': f'Sample thinking process {i} - This includes reasoning steps and thought process.',
            'response_time': np.random.uniform(0.5, 5.0),
            'completion_tokens': np.random.randint(100, 1000),
            'audio_url': audio_urls[i % len(audio_urls)] if i % 3 == 0 else '',
            'image_url': image_urls[i % len(image_urls)] if i % 4 == 0 else '',
            'video_url': video_urls[i % len(video_urls)] if i % 5 == 0 else '',
            'human_label': ''
        })
    
    long_string = """
    # Long content \n## Long content section-1\n### Long content section-1-1 \n 
    - This is a very long string that is used to test.
    - This is the text wrapping behavior in the table cells.
    It should be long enough to exceed the width of the cell and demonstrate how the text is wrapped properly.
    - The text should be formatted correctly with markdown syntax.
    ## Long content section-2\n### Long content section-2-1 \n
    - This is another section of the long string.
    - It should also wrap correctly in the table cell.
    - The content should be readable and formatted nicely.
    """
    data[20]["answer"] = long_string

    return pd.DataFrame(data)

# URL에서 파일명 추출 함수
def extract_filename_from_url(url):
    """URL에서 파일명(확장자 포함)만 추출"""
    if not url:
        return ""
    # URL에서 마지막 '/' 이후의 부분을 가져옴
    filename = url.split('/')[-1]
    # 쿼리 파라미터가 있다면 제거
    if '?' in filename:
        filename = filename.split('?')[0]
    return filename

# 앱 초기화
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# 데이터 로드
df = generate_sample_data()

# 독립변수와 종속변수 정의
independent_vars = ['model', 'prompt_template_name', 'option-1', 'option-2', 'temperature', 'max_tokens']
result_metric_vars = ['response_time', 'completion_tokens']
result_text_vars = ['answer', 'think']
result_url_vars = ['audio_url', 'image_url', 'video_url']
result_vars = result_text_vars + result_url_vars

# 모든 컬럼 목록
all_columns = df.columns.tolist()

# 기본 표시 컬럼 (유용한 컬럼들 미리 선택)
default_columns = ['test_case_id', 'model', 'answer', 'think', 'audio_url', 'image_url', 'video_url', 'response_time']

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
    # 각 컬럼 타입별 너비 설정
    if SHOW_AUDIO_PLAYER_IN_TABLE:
        audio_column_width = '250px'  # 오디오 플레이어를 위한 더 넓은 컬럼
    else:
        audio_column_width = '200px' if VISIBLE_AUDIO_FILE_NAME else '80px'
    
    image_column_width = f'{IMAGE_THUMBNAIL_SIZE + 30}px' if SHOW_IMAGE_THUMBNAILS else '100px'
    
    # 이미지 썸네일 표시 시 행 높이 조정
    row_height = max(TABLE_ROW_HEIGHT, IMAGE_THUMBNAIL_SIZE + 10) if SHOW_IMAGE_THUMBNAILS else TABLE_ROW_HEIGHT
    # 오디오 플레이어 표시 시 행 높이 추가 조정
    if SHOW_AUDIO_PLAYER_IN_TABLE:
        row_height = max(row_height, 60)  # 오디오 플레이어를 위한 최소 높이
    
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
            'height': f'{row_height}px'
        },
        'style_data_conditional': [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {
                    'column_id': result_text_vars,
                },
                'presentation': 'markdown',
                'textAlign': 'left',
                'fontSize': '10px'
            },
            {
                'if': {
                    'column_id': 'audio_url',
                },
                'presentation': 'markdown',
                'textAlign': 'center',
                'width': audio_column_width,
                'maxWidth': audio_column_width,
                'minWidth': audio_column_width
            },
            {
                'if': {
                    'column_id': 'image_url',
                },
                'presentation': 'markdown',
                'textAlign': 'center',
                'width': image_column_width,
                'maxWidth': image_column_width,
                'minWidth': image_column_width,
                'cursor': 'pointer',
                'verticalAlign': 'middle'
            },
            {
                'if': {
                    'column_id': 'video_url',
                },
                'presentation': 'markdown',
                'textAlign': 'center',
                'width': '100px',
                'maxWidth': '100px',
                'minWidth': '100px',
                'cursor': 'pointer'
            }
        ],
        'style_header': {
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'fontSize': '15px',
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
            'height': f'{row_height}px'
        }
    }

# 레이아웃 정의
app.layout = html.Div([
    html.H1("Prompt Test Result Analysis Dashboard", 
            style={'text-align': 'center', 'margin-bottom': '20px'}),
    
    html.Div([
        # 좌측 설정 구역
        html.Div([
            html.H3("Experiment Settings", style={'margin-bottom': '20px'}),
            
            # 비교할 설정 값 선택
            html.Div([
                html.H4("Comparison Condition", style={'margin-bottom': '10px'}),
                
                html.Label("Attribute:"),
                dcc.Dropdown(
                    id='target-var-dropdown',
                    options=[{'label': var, 'value': var} for var in independent_vars],
                    value='model',
                    style={'margin-bottom': '15px'}
                ),
                
                html.Label("Values:"),
                dcc.Dropdown(
                    id='target-values-dropdown',
                    multi=True,
                    style={'margin-bottom': '20px'}
                ),
            ], style={'margin-bottom': '25px', 'padding': '15px', 
                     'border': '1px solid #ddd', 'border-radius': '5px'}),
            
            # 통제 변인 설정
            html.Div([
                html.H4("Filtering Conditions", style={'margin-bottom': '10px'}),
                html.Div(id='control-vars-container'),
            ], style={'margin-bottom': '15px', 'padding': '10px', 
                     'border': '1px solid #ddd', 'border-radius': '5px'}),
            
            # 차트 타입 선택 (조건부 표시)
            html.Div([
                html.H4("Visualization of Metric", style={'margin-bottom': '10px'}),
                                
                html.Label("Variables to Compare:"),
                dcc.Dropdown(
                    id='dependent-var-dropdown',
                    options=[{'label': var, 'value': var} for var in result_metric_vars],
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
            # 테이블 컬럼 선택 (단일 설정으로 통합)
            html.Div([
                html.Label("Columns to show:"),
                dcc.Dropdown(
                    id='table-columns-dropdown',
                    options=[{'label': col, 'value': col} for col in all_columns],
                    value=default_columns,
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
    
    # 미디어 플레이어 모달들
    html.Div(id='audio-modal', style={'display': 'none'}),
    html.Div(id='video-modal', style={'display': 'none'}),
    html.Div(id='image-modal', style={'display': 'none'}),
    
    # 데이터 저장을 위한 숨겨진 div
    html.Div(id='hidden-div', style={'display': 'none'}),
    
    # 통제 변인 값들을 저장하는 숨겨진 store
    dcc.Store(id='control-values-store'),
    
    # 미디어 URL을 저장하는 숨겨진 store들
    dcc.Store(id='audio-data-store'),
    dcc.Store(id='video-data-store'),
    dcc.Store(id='image-data-store')
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
    
    # 미디어 데이터를 위한 딕셔너리들 생성
    audio_data = {}
    video_data = {}
    image_data = {}
    
    # 테이블 컬럼 정의
    columns = []
    for col in selected_columns:
        if col == 'audio_url':
            columns.append({
                'name': '🎵 Audio',
                'id': col,
                'type': 'text',
                'presentation': 'markdown' if SHOW_AUDIO_PLAYER_IN_TABLE else None
            })
        elif col == 'video_url':
            columns.append({
                'name': '🎬 Video',
                'id': col,
                'presentation': 'markdown'
            })
        elif col == 'image_url':
            columns.append({
                'name': '🖼️ Image',
                'id': col,
                'type': 'text',
                'presentation': 'markdown' if SHOW_IMAGE_THUMBNAILS else None
            })
        elif col in result_text_vars:
            columns.append({
                'name': col,
                'id': col,
                'editable': False,
                'type': 'text',
                'presentation': 'markdown'
            })
        elif col == 'human_label':
            columns.append({
                'name': col,
                'id': col,
                'editable': True,
                'type': 'text',
                'presentation': 'markdown'
            })
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric' if col in result_metric_vars else 'text'
            })
    
    # 테이블 데이터 생성
    display_df = data_df.copy()
    
    # 오디오 URL 처리
    if 'audio_url' in display_df.columns:
        for idx, row in display_df.iterrows():
            if row['audio_url']:
                audio_id = f"audio_{table_id_suffix}_{idx}"
                audio_data[audio_id] = row['audio_url']
                
                if SHOW_AUDIO_PLAYER_IN_TABLE:
                    # 테이블에 오디오 플레이어 직접 표시
                    filename = extract_filename_from_url(row['audio_url'])
                    display_df.at[idx, 'audio_url'] = f"<audio controls style='width: 100%; height: 30px;' preload='metadata' title='{filename}'><source src='{row['audio_url']}' />Your browser does not support the audio element.</audio>"
                    
                else:
                    # 기존 방식: 아이콘 또는 파일명 표시
                    if VISIBLE_AUDIO_FILE_NAME:
                        filename = extract_filename_from_url(row['audio_url'])
                        display_df.at[idx, 'audio_url'] = f"🔊 {filename}"
                    else:
                        display_df.at[idx, 'audio_url'] = f"🔊"
            else:
                display_df.at[idx, 'audio_url'] = ""
    
    # 비디오 URL 처리
    if 'video_url' in display_df.columns:
        for idx, row in display_df.iterrows():
            if row['video_url']:
                video_id = f"video_{table_id_suffix}_{idx}"
                video_data[video_id] = row['video_url']
                display_df.at[idx, 'video_url'] = "📹"
            else:
                display_df.at[idx, 'video_url'] = ""
    
    # 이미지 URL 처리
    if 'image_url' in display_df.columns:
        for idx, row in display_df.iterrows():
            if row['image_url']:
                image_id = f"image_{table_id_suffix}_{idx}"
                image_data[image_id] = row['image_url']
                
                if SHOW_IMAGE_THUMBNAILS:
                    # 이미지 썸네일을 마크다운으로 표시
                    display_df.at[idx, 'image_url'] = f'<img src="{row["image_url"]}" width="{IMAGE_THUMBNAIL_SIZE}" height="{IMAGE_THUMBNAIL_SIZE}" style="object-fit: cover; cursor: pointer;" class="image-thumbnail" data-url="{row["image_url"]}" />'
                else:
                    # 아이콘만 표시
                    display_df.at[idx, 'image_url'] = "🖼️ View"
            else:
                display_df.at[idx, 'image_url'] = ""
    
    table_styles = get_table_style()
    
    data_table = dash_table.DataTable(
        id={'type': 'results-table', 'suffix': table_id_suffix},
        columns=columns,
        data=display_df.to_dict('records'),
        editable=True,
        markdown_options={"html": True} if SHOW_AUDIO_PLAYER_IN_TABLE else {},
        filter_action="native",
        sort_action="native",
        page_action="native",
        page_current=0,
        page_size=10,
        **table_styles,
        fixed_rows={'headers': True},
        # 복사 허용 설정
        export_format="csv" if ALLOW_COPY else None,
        export_headers="display" if ALLOW_COPY else None,
        # 컬럼 크기 조절 허용
        style_as_list_view=True,
        css=[{
            'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table',
            'rule': 'table-layout: auto;'
        }, {
            'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th',
            'rule': 'resize: horizontal; overflow: auto;' if ALLOW_COPY else ''
        }],
    )
    
    
    # 미디어 데이터를 포함한 컨테이너 반환
    return html.Div([
        data_table,
        # 미디어 데이터를 숨겨진 div에 저장
        html.Div(
            id=f'audio-data-{table_id_suffix}',
            **{'data-audio-mapping': json.dumps(audio_data)},
            style={'display': 'none'}
        ),
        html.Div(
            id=f'video-data-{table_id_suffix}',
            **{'data-video-mapping': json.dumps(video_data)},
            style={'display': 'none'}
        ),
        html.Div(
            id=f'image-data-{table_id_suffix}',
            **{'data-image-mapping': json.dumps(image_data)},
            style={'display': 'none'}
        )
    ])

# 데이터 필터링 및 테이블 업데이트 - 좌우 분할 비교
@app.callback(
    [Output('comparison-tables-container', 'children'),
     Output('audio-data-store', 'data'),
     Output('video-data-store', 'data'),
     Output('image-data-store', 'data')],
    [Input('target-var-dropdown', 'value'),
     Input('target-values-dropdown', 'value'),
     Input('table-columns-dropdown', 'value'),
     Input('control-values-store', 'data')]
)
def update_comparison_tables(target_var, target_values, selected_columns, control_dict):
    if not target_var or not target_values or not selected_columns:
        return [html.P("Please complete the settings.")], {}, {}, {}
    
    if not control_dict:
        control_dict = {}
    
    # 데이터 필터링
    filtered_df = df.copy()
    
    # 통제 변인 필터링
    for var, value in control_dict.items():
        if value is not None and value != 'any':
            filtered_df = filtered_df[filtered_df[var] == value]
    
    # 미디어 데이터 수집
    audio_data_store = {}
    video_data_store = {}
    image_data_store = {}
    
    # 조건: target_var가 존재하고 target_values가 정확히 2개일 때만 좌우 분할 비교
    if len(target_values) == 2:
        # 실제 데이터프레임에 존재하는 컬럼만 필터링
        columns_to_show = [col for col in selected_columns if col in filtered_df.columns]
        
        # 첫 번째와 두 번째 값으로 분할
        left_value = target_values[0]
        right_value = target_values[1]
        
        left_filtered_df = filtered_df[filtered_df[target_var] == left_value]
        right_filtered_df = filtered_df[filtered_df[target_var] == right_value]
        
        left_display_df = left_filtered_df[columns_to_show]
        right_display_df = right_filtered_df[columns_to_show]
        
        # 미디어 데이터 수집
        for media_type, data_store in [('audio_url', audio_data_store), ('video_url', video_data_store), ('image_url', image_data_store)]:
            if media_type in columns_to_show:
                for idx, row in left_display_df.iterrows():
                    if row[media_type]:
                        media_id = f"{media_type.split('_')[0]}_left_{idx}"
                        data_store[media_id] = row[media_type]
                
                for idx, row in right_display_df.iterrows():
                    if row[media_type]:
                        media_id = f"{media_type.split('_')[0]}_right_{idx}"
                        data_store[media_id] = row[media_type]
        
        tables_content = [
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
                    ], style={'height': f'calc({TABLE_MAX_HEIGHT} + 100px)'})
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
                    ], style={'height': f'calc({TABLE_MAX_HEIGHT} + 100px)'})
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top',
                         'padding': '10px', 'border': '1px solid #ff7f0e', 'border-radius': '5px',
                         'box-sizing': 'border-box'})
            ], style={'width': '100%', 'display': 'block', 'margin-bottom': '30px'})
        ]
        
        return tables_content, audio_data_store, video_data_store, image_data_store
    else:
        # 조건이 맞지 않으면 단일 테이블 표시
        single_filtered_df = filtered_df[filtered_df[target_var].isin(target_values)]
        
        # selected_columns만 사용
        columns_to_show = [col for col in selected_columns if col in single_filtered_df.columns]
        single_display_df = single_filtered_df[columns_to_show]
        
        # 미디어 데이터 수집
        for media_type, data_store in [('audio_url', audio_data_store), ('video_url', video_data_store), ('image_url', image_data_store)]:
            if media_type in columns_to_show:
                for idx, row in single_display_df.iterrows():
                    if row[media_type]:
                        media_id = f"{media_type.split('_')[0]}_single_{idx}"
                        data_store[media_id] = row[media_type]
        
        tables_content = [
            html.H4("Results of Filtered Test Cases"),
            html.P(f"# of Test Cases: {len(single_display_df)}", 
                   style={'font-weight': 'bold', 'margin-bottom': '10px'}),
            html.P("Note: Side-by-side comparison is available when exactly 2 target values are selected.", 
                   style={'color': '#666', 'font-style': 'italic', 'margin-bottom': '15px'}) 
                   if len(target_values) != 2 else None,
            create_data_table(single_display_df, columns_to_show, 'single')
        ]
        
        return tables_content, audio_data_store, video_data_store, image_data_store

# 오디오 셀 클릭 이벤트 처리
@app.callback(
    Output('audio-modal', 'children'),
    Output('audio-modal', 'style'),
    [Input({'type': 'results-table', 'suffix': ALL}, 'active_cell')],
    [State({'type': 'results-table', 'suffix': ALL}, 'data'),
     State({'type': 'results-table', 'suffix': ALL}, 'id'),
     State('audio-data-store', 'data')],
    prevent_initial_call=True
)
def handle_audio_cell_click(active_cells, table_data_list, table_ids, audio_data):
    # 오디오 플레이어가 테이블에 직접 표시되는 경우 모달 비활성화
    if SHOW_AUDIO_PLAYER_IN_TABLE:
        return [], {'display': 'none'}
    
    # 클릭된 테이블과 셀 찾기
    active_cell = None
    table_data = None
    table_suffix = None
    
    for i, cell in enumerate(active_cells):
        if cell and cell.get('column_id') == 'audio_url':
            active_cell = cell
            if i < len(table_data_list):
                table_data = table_data_list[i]
            if i < len(table_ids):
                table_suffix = table_ids[i]['suffix']
            break
    
    if not active_cell or not table_data or not table_suffix:
        return [], {'display': 'none'}
    
    row_index = active_cell['row']
    
    # 해당 행의 오디오 데이터 확인
    if row_index < len(table_data):
        row_data = table_data[row_index]
        audio_cell_value = row_data.get('audio_url', '')
        if audio_cell_value and ('🔊' in audio_cell_value):
            # 저장된 실제 URL 찾기
            audio_id = f"audio_{table_suffix}_{row_index}"
            actual_url = audio_data.get(audio_id)
            
            if actual_url:
                filename = extract_filename_from_url(actual_url)
                
                # 오디오 플레이어 모달 생성
                modal_content = html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Audio Player", style={'margin': '0', 'color': '#333'}),
                            html.Button("✕", 
                                      id={'type': 'close-audio-modal', 'index': f"close_{table_suffix}_{row_index}"}, 
                                      style={'background': 'none', 'border': 'none', 
                                            'font-size': '20px', 'cursor': 'pointer',
                                            'float': 'right', 'color': '#666'})
                        ], style={'display': 'flex', 'justify-content': 'space-between', 
                                 'align-items': 'center', 'margin-bottom': '20px', 
                                 'border-bottom': '1px solid #eee', 'padding-bottom': '10px'}),
                        
                        # 오디오 플레이어
                        html.Audio(
                            src=actual_url,
                            controls=True,
                            autoPlay=False,
                            preload='metadata',
                            style={'width': '100%', 'margin-bottom': '15px'},
                            title=f'Audio file: {filename}'
                        ),
                        
                        # 추가 정보
                        html.Div([
                            html.P(f"File: {filename}", style={'margin': '5px 0', 'color': '#333', 'font-weight': 'bold'}),
                            html.P(f"Row: {row_index + 1}", style={'margin': '5px 0', 'color': '#666'}),
                            html.P(f"Table: {table_suffix.title()}", style={'margin': '5px 0', 'color': '#666'}),
                            html.P(f"URL: {actual_url}", style={'margin': '5px 0', 'color': '#888', 'font-size': '12px', 'word-break': 'break-all'})
                        ])
                        
                    ], style={
                        'background': 'white',
                        'padding': '25px',
                        'border-radius': '10px',
                        'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.15)',
                        'max-width': '600px',
                        'width': '90%',
                        'margin': '10% auto',
                        'position': 'relative'
                    })
                ], style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'background': 'rgba(0, 0, 0, 0.5)',
                    'z-index': '1000',
                    'display': 'flex',
                    'align-items': 'flex-start',
                    'justify-content': 'center',
                    'animation': 'fadeIn 0.3s ease-in-out'
                })
                
                return modal_content, {'display': 'block'}
    
    return [], {'display': 'none'}

# 비디오 셀 클릭 이벤트 처리
@app.callback(
    Output('video-modal', 'children'),
    Output('video-modal', 'style'),
    [Input({'type': 'results-table', 'suffix': ALL}, 'active_cell')],
    [State({'type': 'results-table', 'suffix': ALL}, 'data'),
     State({'type': 'results-table', 'suffix': ALL}, 'id'),
     State('video-data-store', 'data')],
    prevent_initial_call=True
)
def handle_video_cell_click(active_cells, table_data_list, table_ids, video_data):
    
    # 클릭된 테이블과 셀 찾기
    active_cell = None
    table_data = None
    table_suffix = None
    
    for i, cell in enumerate(active_cells):
        if cell and cell.get('column_id') == 'video_url':
            active_cell = cell
            if i < len(table_data_list):
                table_data = table_data_list[i]
            if i < len(table_ids):
                table_suffix = table_ids[i]['suffix']
            break
    
    if not active_cell or not table_data or not table_suffix:
        return [], {'display': 'none'}
    
    row_index = active_cell['row']
    
    # 해당 행의 비디오 데이터 확인
    if row_index < len(table_data):
        row_data = table_data[row_index]
        video_cell_value = row_data.get('video_url', '')
        if video_cell_value and ('📹' in video_cell_value):
            # 저장된 실제 URL 찾기
            video_id = f"video_{table_suffix}_{row_index}"
            actual_url = video_data.get(video_id)
            
            if actual_url:
                filename = extract_filename_from_url(actual_url)
                
                # 비디오 플레이어 컴포넌트
                if DASH_PLAYER_AVAILABLE:
                    video_player = dp.DashPlayer(
                        id=f'video-player-{table_suffix}-{row_index}',
                        url=actual_url,
                        controls=True,
                        width="100%",
                        height="400px",
                        style={'margin-bottom': '15px'}
                    )
                else:
                    # dash_player가 없으면 HTML video 태그 사용
                    video_player = html.Video(
                        src=actual_url,
                        controls=True,
                        style={'width': '100%', 'height': '400px', 'margin-bottom': '15px'}
                    )
                
                # 비디오 플레이어 모달 생성
                modal_content = html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Video Player", style={'margin': '0', 'color': '#333'}),
                            html.Button("✕", 
                                      id={'type': 'close-video-modal', 'index': f"close_{table_suffix}_{row_index}"}, 
                                      style={'background': 'none', 'border': 'none', 
                                            'font-size': '20px', 'cursor': 'pointer',
                                            'float': 'right', 'color': '#666'})
                        ], style={'display': 'flex', 'justify-content': 'space-between', 
                                 'align-items': 'center', 'margin-bottom': '20px', 
                                 'border-bottom': '1px solid #eee', 'padding-bottom': '10px'}),
                        
                        # 비디오 플레이어
                        video_player,
                        
                        # 추가 정보
                        html.Div([
                            html.P(f"File: {filename}", style={'margin': '5px 0', 'color': '#333', 'font-weight': 'bold'}),
                            html.P(f"Row: {row_index + 1}", style={'margin': '5px 0', 'color': '#666'}),
                            html.P(f"Table: {table_suffix.title()}", style={'margin': '5px 0', 'color': '#666'}),
                            html.P(f"URL: {actual_url}", style={'margin': '5px 0', 'color': '#888', 'font-size': '12px', 'word-break': 'break-all'})
                        ])
                        
                    ], style={
                        'background': 'white',
                        'padding': '25px',
                        'border-radius': '10px',
                        'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.15)',
                        'max-width': '800px',
                        'width': '90%',
                        'margin': '5% auto',
                        'position': 'relative'
                    })
                ], style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'background': 'rgba(0, 0, 0, 0.5)',
                    'z-index': '1000',
                    'display': 'flex',
                    'align-items': 'flex-start',
                    'justify-content': 'center',
                    'animation': 'fadeIn 0.3s ease-in-out'
                })
                
                return modal_content, {'display': 'block'}
    
    return [], {'display': 'none'}

# 이미지 셀 클릭 이벤트 처리
@app.callback(
    Output('image-modal', 'children'),
    Output('image-modal', 'style'),
    [Input({'type': 'results-table', 'suffix': ALL}, 'active_cell')],
    [State({'type': 'results-table', 'suffix': ALL}, 'data'),
     State({'type': 'results-table', 'suffix': ALL}, 'id'),
     State('image-data-store', 'data')],
    prevent_initial_call=True
)
def handle_image_cell_click(active_cells, table_data_list, table_ids, image_data):
    
    # 클릭된 테이블과 셀 찾기
    active_cell = None
    table_data = None
    table_suffix = None
    
    for i, cell in enumerate(active_cells):
        if cell and cell.get('column_id') == 'image_url':
            active_cell = cell
            if i < len(table_data_list):
                table_data = table_data_list[i]
            if i < len(table_ids):
                table_suffix = table_ids[i]['suffix']
            break
    
    if not active_cell or not table_data or not table_suffix:
        return [], {'display': 'none'}
    
    row_index = active_cell['row']
    
    # 해당 행의 이미지 데이터 확인
    if row_index < len(table_data):
        row_data = table_data[row_index]
        image_cell_value = row_data.get('image_url', '')
        # 썸네일 또는 아이콘 텍스트 확인
        if image_cell_value and ('img' in image_cell_value or '🖼️' in image_cell_value):
            # 저장된 실제 URL 찾기
            image_id = f"image_{table_suffix}_{row_index}"
            actual_url = image_data.get(image_id)
            
            if actual_url:
                filename = extract_filename_from_url(actual_url)
                
                # 이미지 뷰어 모달 생성
                modal_content = html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Image Viewer", style={'margin': '0', 'color': '#333'}),
                            html.Button("✕", 
                                      id={'type': 'close-image-modal', 'index': f"close_{table_suffix}_{row_index}"}, 
                                      style={'background': 'none', 'border': 'none', 
                                            'font-size': '20px', 'cursor': 'pointer',
                                            'float': 'right', 'color': '#666'})
                        ], style={'display': 'flex', 'justify-content': 'space-between', 
                                 'align-items': 'center', 'margin-bottom': '20px', 
                                 'border-bottom': '1px solid #eee', 'padding-bottom': '10px'}),
                        
                        # 이미지 표시
                        html.Div([
                            html.Img(
                                src=actual_url,
                                style={
                                    'max-width': '100%',
                                    'max-height': '500px',
                                    'width': 'auto',
                                    'height': 'auto',
                                    'border-radius': '8px',
                                    'box-shadow': '0 2px 10px rgba(0,0,0,0.1)'
                                }
                            )
                        ], style={'text-align': 'center', 'margin-bottom': '15px'}),
                        
                        # 추가 정보
                        html.Div([
                            html.P(f"File: {filename}", style={'margin': '5px 0', 'color': '#333', 'font-weight': 'bold'}),
                            html.P(f"Row: {row_index + 1}", style={'margin': '5px 0', 'color': '#666'}),
                            html.P(f"Table: {table_suffix.title()}", style={'margin': '5px 0', 'color': '#666'}),
                            html.P(f"URL: {actual_url}", style={'margin': '5px 0', 'color': '#888', 'font-size': '12px', 'word-break': 'break-all'})
                        ])
                        
                    ], style={
                        'background': 'white',
                        'padding': '25px',
                        'border-radius': '10px',
                        'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.15)',
                        'max-width': '700px',
                        'width': '90%',
                        'margin': '5% auto',
                        'position': 'relative'
                    })
                ], style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'background': 'rgba(0, 0, 0, 0.5)',
                    'z-index': '1000',
                    'display': 'flex',
                    'align-items': 'flex-start',
                    'justify-content': 'center',
                    'animation': 'fadeIn 0.3s ease-in-out'
                })
                
                return modal_content, {'display': 'block'}
    
    return [], {'display': 'none'}

# 모달 닫기 콜백들
@app.callback(
    Output('audio-modal', 'style', allow_duplicate=True),
    Input({'type': 'close-audio-modal', 'index': ALL}, 'n_clicks'),
    State('audio-modal', 'style'),
    prevent_initial_call=True
)
def close_audio_modal(n_clicks_list, current_style):
    if any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list if n_clicks is not None):
        return {'display': 'none'}
    return current_style or {'display': 'block'}

@app.callback(
    Output('video-modal', 'style', allow_duplicate=True),
    Input({'type': 'close-video-modal', 'index': ALL}, 'n_clicks'),
    State('video-modal', 'style'),
    prevent_initial_call=True
)
def close_video_modal(n_clicks_list, current_style):
    if any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list if n_clicks is not None):
        return {'display': 'none'}
    return current_style or {'display': 'block'}

@app.callback(
    Output('image-modal', 'style', allow_duplicate=True),
    Input({'type': 'close-image-modal', 'index': ALL}, 'n_clicks'),
    State('image-modal', 'style'),
    prevent_initial_call=True
)
def close_image_modal(n_clicks_list, current_style):
    if any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list if n_clicks is not None):
        return {'display': 'none'}
    return current_style or {'display': 'block'}

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

# 휴먼 라벨링 저장 콜백
@app.callback(
    Output('hidden-div', 'children'),
    [Input({'type': 'results-table', 'suffix': ALL}, 'data')],
    prevent_initial_call=True
)
def save_human_labels(table_data_list):
    # 여기서 human_label 변경사항을 저장할 수 있습니다
    # 실제 구현에서는 데이터베이스나 파일에 저장하면 됩니다
    return ""

# CSS 스타일 추가
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 테이블 복사 및 선택 설정 */
            .dash-table-container {
                user-select: """ + ("text" if ALLOW_COPY else "none") + """;
                -webkit-user-select: """ + ("text" if ALLOW_COPY else "none") + """;
                -moz-user-select: """ + ("text" if ALLOW_COPY else "none") + """;
                -ms-user-select: """ + ("text" if ALLOW_COPY else "none") + """;
            }
            
            /* 컬럼 크기 조절 가능하게 설정 */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                resize: horizontal;
                overflow: auto;
                min-width: 50px;
                position: relative;
            }
            
            /* 컬럼 크기 조절 핸들 스타일 */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th:hover {
                border-right: 2px solid #007bff;
            }
            
            /* 오디오 플레이어 셀 스타일 */
            .dash-table-container .dash-cell div[data-dash-column="audio_url"] audio {
                width: 100%;
                height: 30px;
                min-width: 200px;
            }
            
            /* 오디오 아이콘/파일명 셀 스타일 (플레이어가 아닐 때) */
            .dash-table-container .dash-cell div[data-dash-column="audio_url"]:not(:has(audio)) {
                cursor: pointer;
                text-align: """ + ("center" if not VISIBLE_AUDIO_FILE_NAME else "left") + """;
                font-size: 16px;
                transition: all 0.2s ease;
                padding: 5px;
                border-radius: 4px;
            }
            
            .dash-table-container .dash-cell div[data-dash-column="audio_url"]:not(:has(audio)):hover {
                transform: scale(1.05);
                background-color: #e3f2fd;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            /* 비디오 셀 스타일 */
            .dash-table-container .dash-cell div[data-dash-column="video_url"] {
                cursor: pointer;
                text-align: center;
                font-size: 16px;
                transition: all 0.2s ease;
                padding: 5px;
                border-radius: 4px;
            }
            
            .dash-table-container .dash-cell div[data-dash-column="video_url"]:hover {
                transform: scale(1.05);
                background-color: #e3f2fd;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            /* 이미지 셀 스타일 */
            .dash-table-container .dash-cell div[data-dash-column="image_url"] {
                cursor: pointer;
                position: relative;
                text-align: center;
            }
            
            /* 이미지 썸네일 스타일 */
            .dash-table-container .dash-cell div[data-dash-column="image_url"] img {
                border-radius: 4px;
                transition: all 0.2s ease;
                display: inline-block;
                vertical-align: middle;
            }
            
            .dash-table-container .dash-cell div[data-dash-column="image_url"] img:hover {
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            
            /* 이미지 호버 프리뷰 */
            .image-hover-preview {
                position: fixed;
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 8px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
                z-index: 999;
                pointer-events: none;
                display: none;
            }
            
            .image-hover-preview img {
                max-width: """ + str(IMAGE_HOVER_PREVIEW_SIZE) + """px;
                max-height: """ + str(IMAGE_HOVER_PREVIEW_SIZE) + """px;
                border-radius: 4px;
            }
            
            /* 복사 가능할 때 추가 스타일 */""" + ("""
            .dash-table-container .dash-cell:hover {
                background-color: #f8f9fa;
            }
            
            .dash-table-container .dash-cell::selection {
                background-color: #007bff;
                color: white;
            }""" if ALLOW_COPY else "") + """
            
            /* 테이블 셀 텍스트 랩핑 개선 */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner *, 
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner *:after, 
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner *:before {
                box-sizing: inherit;
            }
            
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner .dash-spreadsheet-inner table {
                table-layout: auto;
            }
            
            /* 모달 애니메이션 */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .modal-visible {
                animation: fadeIn 0.3s ease-in-out;
            }
            
            /* 복사 가능 표시 */""" + ("""
            .dash-table-container::before {
                content: "📋 Copy enabled - Select and copy text with Ctrl+C";
                position: absolute;
                top: -25px;
                right: 0;
                font-size: 12px;
                color: #6c757d;
                background: #f8f9fa;
                padding: 2px 8px;
                border-radius: 4px;
                border: 1px solid #dee2e6;
            }""" if ALLOW_COPY else "") + """
        </style>
        <script>
            // 이미지 호버 프리뷰 기능
            document.addEventListener('DOMContentLoaded', function() {
                // 호버 프리뷰 컨테이너 생성
                const previewContainer = document.createElement('div');
                previewContainer.className = 'image-hover-preview';
                document.body.appendChild(previewContainer);
                
                // 이미지 썸네일에 호버 이벤트 추가
                document.addEventListener('mouseover', function(e) {
                    const img = e.target;
                    if (img.tagName === 'IMG' && img.classList.contains('image-thumbnail')) {
                        const url = img.getAttribute('data-url') || img.src;
                        previewContainer.innerHTML = '<img src="' + url + '" />';
                        previewContainer.style.display = 'block';
                        
                        // 프리뷰 위치 설정
                        const rect = img.getBoundingClientRect();
                        let left = rect.right + 10;
                        let top = rect.top;
                        
                        // 화면 경계 체크
                        if (left + """ + str(IMAGE_HOVER_PREVIEW_SIZE) + """ > window.innerWidth) {
                            left = rect.left - """ + str(IMAGE_HOVER_PREVIEW_SIZE) + """ - 10;
                        }
                        
                        previewContainer.style.left = left + 'px';
                        previewContainer.style.top = top + 'px';
                    }
                });
                
                document.addEventListener('mouseout', function(e) {
                    if (e.target.tagName === 'IMG' && e.target.classList.contains('image-thumbnail')) {
                        previewContainer.style.display = 'none';
                    }
                });
            });
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''



# 앱 실행
if __name__ == '__main__':
    app.run(debug=True)
