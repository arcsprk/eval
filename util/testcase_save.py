import yaml

class LiteralString(str):
    pass

def literal_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(LiteralString, literal_representer)


def apply_literal_style_recursively(data, keys_to_style):
    """
    딕셔너리나 리스트를 재귀적으로 탐색하며,
    지정된 키의 값을 LiteralString으로 변환합니다.
    """
    if isinstance(data, dict):
        # 입력이 딕셔너리인 경우
        new_dict = {}
        for key, value in data.items():
            if key in keys_to_style:
                # 스타일을 적용할 키를 찾으면 값을 LiteralString으로 변환
                new_dict[key] = LiteralString(str(value))
            else:
                # 그렇지 않으면 하위 구조를 계속 탐색
                new_dict[key] = apply_literal_style_recursively(value, keys_to_style)
        return new_dict
    elif isinstance(data, list):
        # 입력이 리스트인 경우, 각 항목에 대해 재귀 호출
        return [apply_literal_style_recursively(item, keys_to_style) for item in data]
    else:
        # 딕셔너리나 리스트가 아니면 그대로 반환
        return data

def save_to_yaml_nested(dict_test_config: dict, list_text_var: list, file_path: str):
    """
    계층 구조의 딕셔너리를 YAML 파일로 저장합니다.
    list_text_var에 지정된 키의 값은 어느 깊이에 있든 리터럴 블록 스타일(|)로 저장합니다.

    Args:
        dict_test_config (dict): 테스트 설정 정보 딕셔너리 (계층 구조 가능).
        list_text_var (list): 리터럴 블록 스타일로 처리할 키 리스트.
        file_path (str): 저장할 YAML 파일 경로.
    """
    # 재귀 함수를 호출하여 스타일이 적용된 새로운 딕셔너리를 생성
    data_to_save = apply_literal_style_recursively(dict_test_config, list_text_var)
            
    try:
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(data_to_save, yaml_file, allow_unicode=True, sort_keys=False)
        print(f"✅ 설정이 '{file_path}' 파일에 성공적으로 저장되었습니다.")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            print("\n--- 저장된 파일 내용 ---")
            print(f.read())

    except Exception as e:
        print(f"❌ 파일 저장 중 오류가 발생했습니다: {e}")

# --- 함수 사용 예제 ---
if __name__ == "__main__":
  # 1. 계층 구조를 가진 딕셔너리 정의
  dict_nested_config = {
      'test_name': 'Nested Structure Test',
      'environment': 'production',
      'settings': {
          'timeout': 90,
          'credentials': {
              'username': 'prod_user',
              'api_key': 'SECRET-API-KEY-12345' # 스타일 적용 대상 (하위)
          }
      },
      'payloads': [
          {
              'id': 1,
              'query': 'SELECT * FROM users;\n-- This is a comment.' # 스타일 적용 대상 (리스트 내부)
          },
          {
              'id': 2,
              'query': 'SELECT * FROM products;'
          }
      ],
      'api_key': 'TOP-LEVEL-API-KEY' # 스타일 적용 대상 (상위)
  }
  
  # 2. 리터럴 블록으로 처리할 키 리스트 (어느 깊이에 있든 상관 없음)
  list_text_var_nested = ['api_key', 'query']
  
  # 3. 저장할 파일 경로
  output_yaml_path_nested = 'test_config_nested.yaml'
  
  # 4. 함수 호출
  save_to_yaml_nested(dict_nested_config, list_text_var_nested, output_yaml_path_nested)
