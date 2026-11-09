# 일반화 질문·SQL 쌍 생성기

일반화 태그가 붙은 SQL을 먼저 만들고, 각 SQL에 대응하는 영어 질문을 생성한다. 결과는 `id`, `input`(질문), `output`(SQL)을 담은 UTF-8 JSON 배열이다.

## 실행

저장소 루트에서 Python 3.10 이상으로 실행한다. 아래 명령은 seed 42로 원본 SQL당 최대 3개의 질문을 만들어 v1 표본 1,000건을 재현한다.

```shell
python src/data_generation/generalized_pairs/generate.py --count 1000 --seed 42 --max-questions-per-sql 3 --output data/v1/generalized_pairs.json
```

50만 건을 생성할 때는 대용량 비율 파일을 지정하고 원본당 질문 상한은 자동으로 둔다. `data/v1/generated/`의 결과는 Git에서 제외된다.

```shell
python src/data_generation/generalized_pairs/generate.py --count 500000 --seed 42 --sql-types src/data_generation/generalized_pairs/sql_types_large.json --output data/v1/generated/generalized_pairs.json
```

## CLI 옵션

| 옵션 | 필수 여부 | 의미 |
| --- | --- | --- |
| `--count N` | 필수 | 생성할 질문·SQL 쌍의 총수. 0 이상의 정수 |
| `--output 경로` | 필수 | 출력 JSON 경로. 생성에 성공하면 해당 파일을 교체한다. |
| `--seed N` | 선택 | SQL과 질문 표현을 선택하는 난수 시드. 기본값 `0` |
| `--max-questions-per-sql N` | 선택 | 원본 SQL 하나당 서로 다른 질문의 최대 개수. 1 이상의 정수이며 생략하면 자동 |
| `--sql-types 경로` | 선택 | 유형별 비율을 읽을 JSON 경로. 기본값은 생성기 폴더의 `sql_types.json` |

## 유형별 비율

비율은 JSON에서만 지정한다. 기본 파일을 복사해 수정한 뒤 `--sql-types`로 선택할 수 있다.

- `family_percent`는 전체 건수 중 대분류별 백분율이다. 기본값은 일반 조회 50%, 단순 집계 25%, 그룹 조회와 스칼라 서브쿼리 각각 12.5%다.
- `subtype_percent`는 대분류 안에서의 백분율이다. 예를 들어 일반 조회 50%, `where1:=/plain` 8%이면 1,000건 중 40건이 정렬·수량 제한 없이 등호 조건 하나만 있는 조회다. `0`은 해당 유형 제외, `null`은 남은 몫의 자동 배분이다.
- `example_weights`는 일반 레코드(`ordinary`)와 대조쌍 양쪽 레코드(`contrast`)의 상대 가중치다. 기본 설정은 각각 8과 1이다.

## 파일별 역할

| 파일 | 역할 |
| --- | --- |
| [generate.py](generate.py) | CLI 옵션을 읽고 SQL과 질문을 생성해 JSON에 기록한다. |
| [catalog.py](catalog.py) | SQL 구조와 조건·정렬·집계 조합을 정의한다. |
| [contrasts.py](contrasts.py) | 비교 경계나 정렬 방향 등 한 가지 의미가 달라지는 질문·SQL 대조쌍을 만든다. |
| [mappings.json](mappings.json) | SQL 요소를 영어 질문으로 표현하는 문장 틀과 표현 후보를 담는다. |
| [sql_types.py](sql_types.py) | 유형별 비율을 검증하고 생성할 건수를 계산한다. |
| [sql_types.json](sql_types.json) | 일반 조회를 중심으로 한 기본 생성 비율을 지정한다. |
| [sql_types_large.json](sql_types_large.json) | 50만 건 규모에 맞춰 세부 유형의 비율과 대조 가중치를 조정한다. |

## 유의점

- 재현할 때는 seed와 함께 생성 건수, 비율 파일, 질문 상한, 표현 목록도 동일하게 사용한다.
- 비율을 지정하면 생성 전에 원본이 부족한 유형의 서로 다른 질문 표현을 세어 할당량을 확인한다. 대조쌍을 만든 원본도 질문 상한까지 일반 질문을 추가할 수 있다. 용량 오류가 나오면 해당 유형의 비율을 낮추거나 질문 상한을 조정한다.
