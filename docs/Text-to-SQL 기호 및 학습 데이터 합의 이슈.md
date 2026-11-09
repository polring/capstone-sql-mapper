# [설계 합의] Text-to-SQL 기호 규칙과 학습 데이터 형식 확정

**앞모델은 DB 종속적이다.** `shop.db`의 스키마에 맞춰 학습하며, 전처리된 질문만 입력받아 조회 대상과 조건값을 실제 테이블·컬럼에 연결한다. **뒷모델은 DB 독립적이다.** 매퍼가 실제 이름과 값을 `TABLE1`, `COL1`, `#1` 등의 태그로 바꾼 문장을 입력받아 일반화된 SQL을 생성한다.

## 앞모델의 고정 DB 스키마

`SQLite 선언 타입`은 DB에 정의된 타입이고, `태그 타입`은 모델이 사용하는 의미 유형이다. 예를 들어 `orders.order_date`는 DB에서 `TEXT`로 선언하고 모델에서는 `date`로 표시한다. [shop.db](https://github.com/polring/capstone-design-1/blob/feat/%2340-stage3-pipeline/data/stage3/db/shop.db)

| 테이블 | 컬럼 | SQLite 선언 타입 | 태그 타입 | 제약 조건 |
| --- | --- | --- | --- | --- |
| `customers` | `customer_id` | `INTEGER` | `numeric` | 기본키 |
| `customers` | `name` | `TEXT` | `string` | `NOT NULL`, `UNIQUE` |
| `customers` | `city` | `TEXT` | `string` | `NOT NULL` |
| `customers` | `membership` | `TEXT` | `string` | `NOT NULL` |
| `items` | `item_id` | `INTEGER` | `numeric` | 기본키 |
| `items` | `item_name` | `TEXT` | `string` | `NOT NULL`, `UNIQUE` |
| `items` | `category` | `TEXT` | `string` | `NOT NULL` |
| `items` | `price` | `REAL` | `numeric` | `NOT NULL` |
| `items` | `stock` | `INTEGER` | `numeric` | `NOT NULL` |
| `orders` | `order_id` | `INTEGER` | `numeric` | 기본키 |
| `orders` | `customer_id` | `INTEGER` | `numeric` | `NOT NULL`, 외래키 → `customers.customer_id` |
| `orders` | `item_id` | `INTEGER` | `numeric` | `NOT NULL`, 외래키 → `items.item_id` |
| `orders` | `quantity` | `INTEGER` | `numeric` | `NOT NULL` |
| `orders` | `status` | `TEXT` | `string` | `NOT NULL` |
| `orders` | `order_date` | `TEXT` | `date` | `NOT NULL` |

## 학습할 SQL 표현

최종 SQL은 `shop.db` 스키마에 선언된 컬럼명을 테이블 접두어 없이 사용한다. 예를 들어 앞모델 태그와 매퍼 기록에서 `items.item_name`으로 식별한 컬럼은 후처리기가 `item_name`으로 출력한다. 이 규칙은 서브쿼리 안에서도 동일하게 적용한다.

앞모델과 뒷모델은 **같은 SQL 범위**에 해당하는 질문으로 학습한다. 앞모델은 질문의 구간·스키마 연결을, 뒷모델은 일반화 SQL을 정답으로 사용한다. 두 모델의 학습 데이터는 아래 문법과 조합을 모두 포함한다.

| 영역 | 학습할 표현과 범위 |
| --- | --- |
| 조회 | 단일 테이블 `SELECT`의 컬럼 하나 또는 `*`; 그룹 조회에서는 그룹 키 하나와 집계식 하나 |
| 일반 조회의 `WHERE` | `=`, `!=`, `<`, `>`, `<=`, `>=`, `IN`, `BETWEEN`, `LIKE`로 조건을 최대 3개까지 쓴다. 서로 다른 컬럼의 조건은 모두 `AND`로 잇거나 모두 `OR`로 잇는다. `IN`의 값 목록과 `BETWEEN`의 범위는 각각 조건 하나로 센다. |
| 집계·그룹·서브쿼리의 `WHERE` | 여러 조건은 `AND`로만 잇는다. 일반 행 필터는 최대 2개이며, 스칼라 서브쿼리에서는 바깥쪽과 안쪽에 이 상한을 각각 적용한다. |
| 정렬·건수 | `ORDER BY`의 `ASC`·`DESC`, `LIMIT`을 지원한다. 일반 조회와 그룹 조회 모두 사용자가 동점 순서를 명시한 경우에만 요청한 기준을 보조 정렬 키로 추가한다. |
| 날짜 | 질문에 명시된 날짜의 등호·전후·범위 조건과 문자열 형태의 날짜 리터럴 |
| 집계 | 행 수는 `COUNT(*)`로 센다. 뒷모델의 일반화 SQL에서 `AVG`·`SUM`의 인자는 `numeric` 컬럼 태그이고, `MIN`·`MAX`의 인자는 `numeric` 또는 `date` 컬럼 태그다. 집계식은 하나만 사용하며, `GROUP BY`의 그룹 키는 하나까지 허용한다. |
| `HAVING` | `GROUP BY`가 있는 질의에서만 사용한다. `SELECT`에 쓴 숫자 집계식을 그대로 반복하고, `=`, `!=`, `<`, `>`, `<=`, `>=` 중 하나로 정수 또는 소수와 비교하는 조건 하나를 허용한다. |
| 서브쿼리 | 깊이 1의 비상관 스칼라 서브쿼리에서 바깥쪽 `numeric` 컬럼을 같은 테이블·같은 컬럼의 `AVG`·`MIN`·`MAX` 결과와 비교한다. 안팎의 컬럼 태그는 같은 번호를 쓴다. |

| 영역 | 이번 학습 범위에서 제외할 표현 |
| --- | --- |
| 조건 | 같은 `WHERE`에서 `AND`와 `OR`를 섞는 조건식, 집계·그룹·서브쿼리의 `WHERE`에서 조건을 잇는 `OR`, `IS NULL`·`IS NOT NULL` 조건식 |
| 조회·결과 | `JOIN`, `DISTINCT`, `OFFSET`, `EXISTS`, `CASE` |
| 집계 | `COUNT(컬럼명)`, `COUNT(DISTINCT 컬럼명)`, `string` 컬럼 태그를 인자로 쓰는 집계식, 여러 집계식·그룹 키 |
| `HAVING` | 날짜·문자열 비교, 여러 집계 조건, `SELECT`와 다른 집계식을 사용하는 조건 |
| 서브쿼리 | 상관·중첩 서브쿼리 |
| 날짜·시간 | “지난달 주문”처럼 기준 날짜가 필요한 상대 날짜 표현, 시각·일시·시간대·기간 조건 |

`shop.db` 예시에서 집계 인자로 쓰는 실제 컬럼은 `items.price`, `items.stock`, `orders.quantity`이며, `MIN`·`MAX`에는 `orders.order_date`도 사용한다. 이 목록은 일반화된 예시를 `shop.db`에 매핑할 때 적용한다. 뒷모델은 `AVG(<TABLE1.COL1, numeric>)`처럼 질문 안에서 부여한 번호와 타입으로 집계식을 표현한다.

예를 들어 `WHERE category IN ('kitchen', 'sports') OR price < 10.00`은 범주 조건과 가격 조건을 `OR`로 연결한 일반 조회다. 조건에 맞는 행이 없어 `AVG`의 결과가 `NULL`인 경우도 유효한 학습 예시로 인정한다.

“주문 수가 많은 고객 20명”의 정렬·건수 절은 `ORDER BY COUNT(*) DESC LIMIT 20`이다. “동점이면 고객 ID가 작은 순”까지 요청하면 `ORDER BY COUNT(*) DESC, customer_id ASC LIMIT 20`을 쓴다.

날짜의 전후·범위 의미와 집계·그룹·서브쿼리에서 각 조건이 속하는 SQL 범위를 두 모델의 학습·평가에 반영한다.

## DB 독립적인 태그 유형과 값 처리

뒷모델은 실제 테이블·컬럼 이름을 보지 않고, 컬럼의 의미 유형을 **`string`·`numeric`·`date` 세 가지**로 일반화한다. 이 세 유형으로 표현할 수 있는 DB 스키마에 재사용하는 범위이며, 모든 DB 자료형을 지원한다는 뜻은 아니다. SQL 출력·복원 방언은 SQLite를 기준으로 한다. 날짜는 DB에 따라 문자나 숫자로 저장될 수 있어 실제 저장 형식과 태그 유형을 구분한다. [SQLite 자료형](https://www.sqlite.org/datatype3.html)

| 의미 유형 | 태그 타입 | 값·연산 처리 |
| --- | --- | --- |
| 문자 | `string` | 정확한 문자열, 대소문자·정렬 규칙, `LIKE` 패턴과 이스케이프 |
| 숫자 | `numeric` | 정수·소수·실수를 한 유형으로 묶는다. 비교·집계 구조는 뒷모델이 정하고, 정수 전용 위치·정밀도·단위·리터럴 표기는 후처리기가 검증한다. |
| 날짜 | `date` | 매퍼는 날짜 원문(예: `March 5, 2025`)을 보관한다. 뒷모델은 전후·당일·범위에 맞는 조건식을 생성하고 날짜 태그를 작은따옴표로 감싼다. 후처리기는 태그 자리에 `orders.order_date`의 저장 형식에 맞춘 값(예: `2025-03-05`)을 넣는다. |

불리언, 시각·일시·기간, 이진·JSON·배열·공간 값, 사용자 정의 타입은 이번 태그와 SQL 학습 범위에서 제외한다. 지원되지 않는 컬럼을 임의로 `string`이나 `numeric`으로 바꾸어 처리하지 않는다.

| 값·표현의 상황 | 공통 규칙 |
| --- | --- |
| 값 여러 개 | 추론 시 앞모델은 같은 컬럼의 값에 질문 등장 순서대로 `#1`~`#3`을 붙인다. 매퍼는 각 값 번호를 유지하고 원문 값을 기록한다. `IN` 목록은 그 순서대로 복원한다. |
| `BETWEEN` 범위 | 뒷모델은 두 경계의 태그를 질문 순서대로 출력한다. 후처리기는 두 값을 숫자 또는 날짜로 해석하고 하한, 상한 순서로 복원한다. 예를 들어 “재고가 80개와 50개 사이”이면 매퍼 기록은 `#1 = 80`, `#2 = 50`이고 최종 조건은 `stock BETWEEN 50 AND 80`이다. |
| `LIKE` 검색 조각 | 검색어만 다른 조건값처럼 `[...]`로 표시하고 매퍼가 원문을 보관한다. 뒷모델은 접두·접미·포함 의미에 따라 SQL의 `%` 위치를 정한다. 검색어 자체의 `%`·`_`는 와일드카드와 구분해 후처리기에서 이스케이프한다. |
| `LIMIT` | 수량을 질문에 그대로 남기고 뒷모델이 정수를 직접 생성한다. 예를 들어 “상위 20개”는 `LIMIT 20`으로 출력한다. 태그 번호 상한은 이 수량에 적용되지 않는다. |
| 상품 가격 | `items.price`의 단위는 달러다. 입력 가격의 소수 자릿수 처리와 최종 SQL의 숫자 표기는 앞모델·매퍼·후처리기 담당자가 정한다. |
| 재고·주문 수량 | `items.stock`은 재고 수량, `orders.quantity`는 주문 수량이며 둘 다 개 단위다. 후처리기는 수량 조건값을 정수로 표기한다. “재고 10개 미만”은 `stock < 10`으로 복원한다. |
| 값 정규화 | 질문에서 사용한 별칭과 DB 저장값이 다르면 대응 관계를 기록한다. |
| 문자열 리터럴 | 뒷모델이 값 태그를 SQL 작은따옴표로 감싼다. 후처리기는 원문 값 안의 작은따옴표 하나(`'`)를 두 개(`''`)로 이스케이프한 뒤 태그 부분만 치환한다. |

지원 태그 유형은 이 세 가지로 고정한다. 다른 유형이 필요한 질문은 지원 범위 밖으로 판정한다.

예를 들어 고객 이름이 `O'Brien`이면 뒷모델은 `WHERE <TABLE1.COL1, string> = '[TABLE1.COL1, string, #1]'`을 출력한다. 후처리기가 태그 자리에 `O''Brien`을 넣으면 최종 조건은 `WHERE name = 'O''Brien'`이 된다.

### `LIKE` 값 표시 예시

고정 DB에 `items.item_name`이 있다고 가정한다. `cam`만 조건값 태그로 바꾸고, `ends with`는 검색 방식의 단서로 문장에 남긴다.

| 단계 | 결과 |
| --- | --- |
| 전처리된 질문 | `Show item names where the name ends with cam.` |
| 앞모델 표시 | `Show {items.item_name, string} where the <items.item_name, string> ends with [items.item_name, string, #1].` |
| 매퍼 출력 | `Show {TABLE1.COL1, string} where the <TABLE1.COL1, string> ends with [TABLE1.COL1, string, #1].` |
| 매퍼 기록 | `<TABLE1> → items`, `<TABLE1.COL1, string> → items.item_name`, `[TABLE1.COL1, string, #1] → cam` |
| 뒷모델 출력 | `SELECT {TABLE1.COL1, string} FROM <TABLE1> WHERE <TABLE1.COL1, string> LIKE '%[TABLE1.COL1, string, #1]'` |
| 최종 SQL | `SELECT item_name FROM items WHERE item_name LIKE '%cam'` |

`starts with cam`이면 `LIKE '[TABLE1.COL1, string, #1]%'`, `contains cam`이면 `LIKE '%[TABLE1.COL1, string, #1]%'`로 출력한다.

검색어가 `%cam`이면 매퍼는 `%cam` 전체를 조건값 하나로 보관한다. 후처리기는 태그 자리에 값을 넣을 때 값 안의 `%`, `_`, 이스케이프 문자 자체를 이스케이프하고, 태그 밖의 와일드카드는 유지한다. 예를 들어 “`%cam`으로 끝나는 이름”은 `!`를 이스케이프 문자로 쓸 경우 `LIKE '%!%cam' ESCAPE '!'`로 복원된다.

## 기호 규칙

테이블·컬럼 태그 `<...>`에서는 기본키·외래키 표기를 생략한다.

| 표시 | 뜻 | 실제 이름을 사용한 예시 | 뒷모델용 일반화 예시 |
| --- | --- | --- | --- |
| `<...>` | 테이블 또는 일반 컬럼 | `<items>`, `<items.category, string>` | `<TABLE1>`, `<TABLE1.COL2, string>` |
| `{...}` | 질문에서는 최종 조회에 쓰일 컬럼 또는 `*`, 일반화 SQL에서는 최종 `SELECT`의 출력식 하나 | `{items.price, numeric}`, `{*}` | `{TABLE1.COL1, numeric}`, `{AVG(<TABLE1.COL1, numeric>)}`, `{COUNT(*)}`, `{*}` |
| `[컬럼, 타입, #n]` | 컬럼과 비교하는 조건값 | `[items.category, string, #1]` | `[TABLE1.COL2, string, #1]` |
| `[AGG_VALUE, numeric, #1]` | `HAVING`에서 집계 결과와 비교하는 숫자 | `[AGG_VALUE, numeric, #1]` | `[AGG_VALUE, numeric, #1]` |

- 컬럼과 조건값 표시에는 위 의미 유형을 반드시 쓴다. 예: `<TABLE1.COL1, numeric>`, `[TABLE1.COL1, numeric, #1]`, `<TABLE1.COL2, date>`. 테이블 표시와 `{*}`에는 자료형을 쓰지 않는다.
- 추론 시 매퍼는 `TABLEn`과 각 테이블의 `COLn`을 질문에서 해당 테이블·컬럼이 처음 등장하는 순서대로 각각 `1`부터 부여한다. 컬럼 태그에만 나타난 테이블도 등장한 것으로 센다. 같은 실제 테이블·컬럼이 다시 나오면 기존 번호를 재사용한다. `shop.db`는 실제 이름과 자료형의 근거이며, `TABLEn`·`COLn` 번호를 저장하지 않는다.
- `TABLEn`은 최대 `3`, 각 테이블의 `COLn`은 최대 `6`까지 사용한다. 따라서 현재 학습 데이터는 한 질문에서 서로 다른 테이블 최대 3개, 테이블당 서로 다른 컬럼 최대 6개로 제한한다. 다만 위 SQL 학습 범위는 단일 테이블이므로 현재 데이터에서는 `TABLE1`만 사용한다. 이 상한을 넘는 질문은 현재 지원 범위 밖이다.
- 앞모델은 조건값이 하나뿐이어도 `#1`을 붙인다. 같은 컬럼의 값이 여러 번 나오면 질문에 나온 순서대로 `[items.category, string, #1]`, `[items.category, string, #2]`처럼 구분하고 `#3`까지만 허용한다. 같은 컬럼에 네 번째 값이 필요한 질문은 현재 지원 범위 밖이다.
- 뒷모델은 문자열·날짜 태그를 작은따옴표 안에 쓰고 숫자 태그는 그대로 쓴다. 후처리기는 각 태그를 해당 타입에 맞게 처리한 값으로 치환한다. `LIKE` 패턴의 `%` 위치와 `LIMIT` 수량은 뒷모델이 정한다.

### 질문의 태그에서 SQL의 태그로 변환하는 규칙

앞모델은 테이블·컬럼·값에 해당하는 구간을 표시하고, `for each`, `count`, `average`, `at least`, `above`처럼 그룹·집계·비교를 결정하는 표현은 질문에 남긴다. 뒷모델은 **SQL에서의 역할에 맞춰 `{...}`와 `<...>`를 생성한다.** 최종 `SELECT`의 각 출력식은 `{...}`로 감싸고, `FROM`의 테이블이나 함수 인자·조건·그룹 키에 쓰이는 컬럼은 `<...>`로 표시한다. 서브쿼리에서 비교용으로 계산한 식에는 최종 출력 표시를 붙이지 않는다.

입력에 `{...}`가 없어도 행 수를 묻는 질문이라면 `{COUNT(*)}`를 생성한다. 독립된 `<TABLE1>` 태그가 없어도 `{TABLE1.COL1, numeric}`에 들어 있는 테이블 번호로 `FROM <TABLE1>`을 생성할 수 있다. 이때 SQL에 쓰는 테이블·컬럼 번호와 값 태그는 모두 해당 질문의 매퍼 기록으로 복원할 수 있어야 한다.

**1. 상품별 가격 조회**

| 단계 | 결과 |
| --- | --- |
| 질문 | `Show the price of each item.` |
| 앞모델 표시 | `Show the {items.price, numeric} of each <items>.` |
| 매퍼 출력 | `Show the {TABLE1.COL1, numeric} of each <TABLE1>.` |
| 뒷모델 정답 | `SELECT {TABLE1.COL1, numeric} FROM <TABLE1>` |
| 최종 SQL | `SELECT price FROM items` |

**2. 전체 상품의 평균 가격 조회**

| 단계 | 결과 |
| --- | --- |
| 질문 | `Show the average price of all items.` |
| 앞모델 표시 | `Show the average {items.price, numeric} of all <items>.` |
| 매퍼 출력 | `Show the average {TABLE1.COL1, numeric} of all <TABLE1>.` |
| 뒷모델 정답 | `SELECT {AVG(<TABLE1.COL1, numeric>)} FROM <TABLE1>` |
| 최종 SQL | `SELECT AVG(price) FROM items` |

**3. 전체 주문 건수 조회**

| 단계 | 결과 |
| --- | --- |
| 질문 | `How many orders are there?` |
| 앞모델 표시 | `How many <orders> are there?` |
| 매퍼 출력 | `How many <TABLE1> are there?` |
| 뒷모델 정답 | `SELECT {COUNT(*)} FROM <TABLE1>` |
| 최종 SQL | `SELECT COUNT(*) FROM orders` |

**4. 주문이 3건 이상인 고객의 ID와 주문 건수 조회**

| 단계 | 결과 |
| --- | --- |
| 질문 | `For each customer ID in orders, show the ID and order count when that count is at least 3.` |
| 앞모델 표시 | `For each <orders.customer_id, numeric> in <orders>, show the {orders.customer_id, numeric} and order count when that count is at least [AGG_VALUE, numeric, #1].` |
| 매퍼 출력 | `For each <TABLE1.COL1, numeric> in <TABLE1>, show the {TABLE1.COL1, numeric} and order count when that count is at least [AGG_VALUE, numeric, #1].` |
| 뒷모델 정답 | `SELECT {TABLE1.COL1, numeric}, {COUNT(*)} FROM <TABLE1> GROUP BY <TABLE1.COL1, numeric> HAVING COUNT(*) >= [AGG_VALUE, numeric, #1]` |
| 최종 SQL | `SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) >= 3` |

매퍼는 `TABLE1`을 `orders`에, `TABLE1.COL1`을 `orders.customer_id`에, `[AGG_VALUE, numeric, #1]`을 `3`에 연결한다. 뒷모델은 `SELECT`에서 출력할 주문 건수를 `{COUNT(*)}`로 표시하고, `HAVING`에서는 같은 식을 비교에 사용하므로 `COUNT(*)`로 쓴다. `HAVING` 비교값은 질문당 하나이므로 `#1`을 쓴다.

### 서브쿼리에서 평균 가격을 비교에 사용하는 경우

| 단계 | 결과 |
| --- | --- |
| 질문 | `Show item names where the price is above the average price.` |
| 매퍼 출력 | `Show {TABLE1.COL1, string} where the <TABLE1.COL2, numeric> is above the average <TABLE1.COL2, numeric>.` |
| 뒷모델 정답 | `SELECT {TABLE1.COL1, string} FROM <TABLE1> WHERE <TABLE1.COL2, numeric> > (SELECT AVG(<TABLE1.COL2, numeric>) FROM <TABLE1>)` |
| 최종 SQL | `SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items)` |

두 `price`는 모두 `items.price`이므로 같은 `COL2`를 쓴다. `above the average`는 문장에 남기고, 평균 가격은 서브쿼리에서 계산한다. 평균을 구할 대상이 지정된 질문에서는 그 범위를 나타내는 표현도 보존한다. 예를 들어 “주방 상품 중 **전체 상품**의 평균 가격보다 비싼 것”은 바깥 조회만 주방 상품으로 제한하고, “주방 상품 중 **주방 상품**의 평균 가격보다 비싼 것”은 평균을 구하는 안쪽 조회에도 같은 조건을 건다.

## 동일 질문의 학습 데이터 예시

### 1. 기존 모델(원본 질문과 정답 SQL)의 학습 데이터

아래 JSON은 [`feat/#40-stage3-pipeline`의 기존 모델 학습 데이터](https://github.com/polring/capstone-design-1/blob/feat/%2340-stage3-pipeline/data/stage3/train/pairs.json.gz) 중 한 건이다.

```json
{
  "question": "what are the names of the things from the kitchen category?",
  "sql": "SELECT item_name FROM items WHERE category = 'kitchen'",
  "sql_group": "SELECT item_name FROM items WHERE category = 'kitchen'",
  "source": "composer"
}
```

### 2. 앞모델의 정답 형식

전처리기는 테이블·컬럼 표현을 정규화하고 조건값의 원문을 보존한다. 이 예시에서는 질문을 그대로 전달한다. 구간 위치는 앞모델과 매퍼가 공유하는 전처리 결과를 기준으로 계산한다.

앞모델 학습 데이터는 `input`·`output` 키를 사용한다. 구간·라벨 방식의 `output`은 라벨 객체다. `start`는 포함하고 `end`는 포함하지 않는 **0부터 시작하는 문자 위치**다. `select`는 여러 조회 대상을 위해 배열로 둔다.

```json
{
  "input": "what are the names of the things from the kitchen category?",
  "output": {
    "table": "<items>",
    "select": [
      {"mark": "{items.item_name, string}", "start": 13, "end": 18}
    ],
    "column_mentions": [
      {"mark": "<items.category, string>", "start": 50, "end": 58}
    ],
    "values": [
      {"mark": "[items.category, string, #1]", "start": 42, "end": 49}
    ]
  }
}
```

이 예시에서 구간은 각각 `names`, `category`, `kitchen`이다. 앞모델은 구간과 고정 DB 컬럼의 연결을 예측하고, 매퍼가 표시 문장을 조립한다.

**또는** 앞모델이 태그가 들어간 문장 전체를 생성할 수 있다. 이 경우 `output`은 문자열이며, 입력의 태그 밖 문장과 값 구간을 원문에 다시 맞춰야 한다.

```json
{
  "input": "what are the names of the things from the kitchen category?",
  "output": "what are the {items.item_name, string} of the things from the [items.category, string, #1] <items.category, string>?"
}
```

두 예시는 앞모델이 매퍼에 전달할 출력 형식의 후보다.

### 3. 매퍼의 출력과 복원 기록

매퍼는 실제 테이블·컬럼에 일반화 번호를 부여하고, 앞모델이 붙인 값 번호를 유지한다. 이 예시에서는 일반화된 값 태그 `[TABLE1.COL2, string, #1]` 전체를 키로 삼아 원문 값 `kitchen`을 기록한다.

```text
what are the {TABLE1.COL1, string} of the things from the [TABLE1.COL2, string, #1] <TABLE1.COL2, string>?
```

| 실제 표시 | 일반화된 표시 | 대응하는 테이블·컬럼·값 |
| --- | --- | --- |
| `<items>` | `<TABLE1>` | `items` |
| `{items.item_name, string}` | `{TABLE1.COL1, string}` | `items.item_name` |
| `<items.category, string>` | `<TABLE1.COL2, string>` | `items.category` |
| `[items.category, string, #1]` | `[TABLE1.COL2, string, #1]` | `kitchen` |

### 4. 뒷모델의 학습 데이터

```json
{
  "input": "what are the {TABLE1.COL1, string} of the things from the [TABLE1.COL2, string, #1] <TABLE1.COL2, string>?",
  "output": "SELECT {TABLE1.COL1, string} FROM <TABLE1> WHERE <TABLE1.COL2, string> = '[TABLE1.COL2, string, #1]'"
}
```

후처리기가 매퍼의 기록을 적용하면 다음 최종 SQL을 얻는다.

```sql
SELECT item_name FROM items WHERE category = 'kitchen'
```

## 앞모델과 매퍼가 합의할 항목

- [ ] **출력 형식과 원문 값의 대응.** 앞모델이 매퍼에 구간·라벨 객체를 전달할지, 태그 문장을 전달할지 정한다. 어느 형식이든 같은 단어가 반복된 질문에서 매퍼가 각 값 태그에 해당하는 원문 구간을 식별할 수 있어야 한다.
- [ ] **태그로 바꿀 구간.** `cheapest`처럼 한 단어가 가격 컬럼과 오름차순을 함께 뜻하는 경우, 앞모델이 컬럼을 어떻게 표시하고 매퍼가 일반화 문장에 정렬 단서를 어떻게 남길지 정한다.

## 담당자별로 정할 항목

각 담당자는 합의한 입력·출력 형식과 기호 규칙에 맞춰 다음 구현 방식을 선택한다.

| 담당자 | 개별 결정 사항 |
| --- | --- |
| 앞모델 | 전처리된 질문을 어떻게 토큰화할지, 어떤 모델 구조와 학습 방법으로 조회 대상·컬럼·값 구간을 찾을지 정한다. 모델 내부의 예측을 매퍼와 합의한 출력 형식으로 변환하는 방법도 선택한다. |
| 매퍼 | `TABLEn`과 `COLn`을 첫 등장 순서대로 부여하고, 같은 컬럼이 다시 나오면 기존 번호를 찾는 데 사용할 자료구조를 정한다. 태그로 바꾸기 전의 문자 위치를 보존하면서 일반화 문장을 조립하고, 복원 기록의 값이 원문 구간과 일치하는지 검사하는 방법도 선택한다. |
| 뒷모델 | 일반화 태그 전체를 한 토큰으로 처리할지 구성 요소로 나눌지 정한다. 일반화된 질문을 SQL로 바꾸는 모델 구조·학습 방법과, 출력 토큰에서 합의한 표기의 SQL을 생성하는 방법도 선택한다. |

## 학습 예시 검증

질문 한 건에 대해 앞모델의 출력, 매퍼의 일반화 문장과 복원 기록, 뒷모델의 SQL, 최종 SQL이 서로 대응하는지 확인한다. 이때 원문 구간, 태그 번호, 날짜 리터럴, `LIMIT` 수량을 대조한다.

동점 순서를 지정하지 않은 질문의 실행 결과를 검증할 때는 동점 행끼리 순서가 바뀌어도 허용한다. `LIMIT` 경계에 동점 행이 여럿이면 그중 어느 행이 남은 자리에 포함돼도 허용한다.

## 모델과 후처리기의 역할 분담

질문 표현을 컬럼에 연결하거나 `at least`를 `>=`로 해석하는 판단은 모델이 학습 예시를 통해 익히도록 한다. 매핑 기록과 스키마에 일정한 규칙을 적용해 처리할 수 있는 일은 가능한 한 후처리기에 맡긴다. 예를 들어 뒷모델은 가격 범위를 `BETWEEN`으로 표현하고, 후처리기는 기록된 두 가격을 하한·상한 순서로 복원한다.
