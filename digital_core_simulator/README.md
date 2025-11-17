# 🪨 디지털 코어 시뮬레이터 (Digital Core Simulator)

암석물성계측 실습을 위한 인터랙티브 3D 웹 기반 시뮬레이터

## 📋 프로젝트 개요

이 프로젝트는 암석물성계측 실습 과목을 위한 디지털 트윈 도구입니다. 학생들이 다양한 암석 타입의 물성값을 시각적으로 이해하고, 파라미터 변화에 따른 물성 변화를 실시간으로 관찰할 수 있습니다.

### 주요 기능

- **3D 디지털 코어 모델**: Three.js를 사용한 실시간 3D 시각화
- **미세구조 단면 시각화**: 공극률에 따른 암석 내부 구조 표현
- **인터랙티브 물성 조절**: 슬라이더로 공극률, 밀도, 탄성파속도 실시간 조절
- **응력-변형률 곡선**: 암석 타입별 역학적 거동 시뮬레이션
- **상관관계 분석**: 공극률-투수율, 탄성파속도-공극률 관계 시각화
- **6가지 암석 타입**: 화강암, 사암, 석회암, 셰일, 현무암, 대리석

### 측정/분석 가능한 물성값

1. **밀도 (Density)**: g/cm³
2. **공극률 (Porosity)**: %
3. **투수율 (Permeability)**: mD (millidarcy)
4. **P파 속도 (Vp)**: m/s
5. **S파 속도 (Vs)**: m/s
6. **일축압축강도 (UCS)**: MPa
7. **인장강도 (Tensile Strength)**: MPa
8. **영률 (Young's Modulus)**: GPa
9. **포아송비 (Poisson's Ratio)**: 무차원
10. **Vp/Vs 비**: 무차원

## 🚀 설치 및 실행 방법

### 1. 필요 조건

- Python 3.7 이상
- 웹 브라우저 (Chrome, Firefox, Safari, Edge 등)
- numpy 패키지

### 2. 샘플 데이터 생성

```bash
cd digital_core_simulator
python generate_rock_data.py
```

실행 후 `data/rock_properties_data.json` 파일이 생성됩니다.

### 3. 웹 서버 실행

파일 시스템 보안 제한으로 인해 로컬 웹 서버를 통해 실행해야 합니다.

**방법 1: Python 내장 웹 서버**
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

**방법 2: Node.js http-server**
```bash
npx http-server -p 8000
```

**방법 3: VS Code Live Server 확장**
- VS Code에서 `index.html` 우클릭
- "Open with Live Server" 선택

### 4. 브라우저에서 실행

웹 브라우저에서 다음 주소로 접속:
```
http://localhost:8000
```

## 📖 사용 방법

### 기본 사용법

1. **암석 타입 선택**
   - 왼쪽 제어 패널에서 암석 타입을 선택합니다
   - 화강암(Granite), 사암(Sandstone), 석회암(Limestone), 셰일(Shale), 현무암(Basalt), 대리석(Marble) 중 선택

2. **샘플 선택**
   - 각 암석 타입마다 15개의 샘플이 자동 생성됩니다
   - 샘플을 클릭하면 해당 샘플의 물성값이 적용됩니다

3. **물성 조절**
   - 공극률, 밀도, P파 속도를 슬라이더로 조절할 수 있습니다
   - 조절 즉시 3D 모델과 차트가 업데이트됩니다

4. **3D 모델 조작**
   - 마우스 드래그: 회전
   - 마우스 휠: 확대/축소
   - 우클릭 드래그: 이동

### 고급 기능

#### 응력-변형률 곡선 분석
- 각 암석 타입의 역학적 거동을 시각화
- 탄성 구간, 소성 구간, 파괴 구간을 확인 가능
- 최대 응력(Peak Stress)과 최대 변형률(Peak Strain) 표시

#### 공극률-투수율 관계
- Kozeny-Carman 관계식 기반 상관관계
- 로그 스케일로 표시되어 넓은 범위의 투수율 관찰 가능

#### 탄성파속도-공극률 관계
- Wyllie time-average equation 기반
- 공극률 증가에 따른 P파 속도 감소 경향 확인

## 📊 데이터 구조

### rock_properties_data.json

```json
{
  "metadata": {
    "generated_at": "ISO 8601 timestamp",
    "description": "암석물성계측 실습용 샘플 데이터",
    "rock_types": ["Granite", "Sandstone", ...],
    "properties": [...],
    "units": {...}
  },
  "rock_type_info": {
    "Granite": {
      "density": [min, max],
      "porosity": [min, max],
      ...
    }
  },
  "samples": {
    "Granite": [
      {
        "sample_id": "Granite_001",
        "density": 2.65,
        "porosity": 1.2,
        ...
      }
    ]
  },
  "stress_strain_curves": {...},
  "correlations": {...}
}
```

## 🎓 교육적 활용

### 수업에서 활용하기

1. **물성 측정 실습과 연계**
   - 학생들이 실험실에서 측정한 실제 데이터와 비교
   - 측정값이 시뮬레이터의 범위 내에 있는지 확인

2. **암석 타입별 특성 이해**
   - 각 암석 타입의 물성 범위 학습
   - 화강암(저공극률, 고강도)과 사암(고공극률, 저강도)의 차이 이해

3. **상관관계 분석**
   - 공극률과 다른 물성값의 관계 이해
   - 경험식(Kozeny-Carman, Wyllie equation) 학습

4. **디지털 트윈 개념 소개**
   - 실제 암석 코어를 디지털로 모델링하는 과정 이해
   - 시뮬레이션을 통한 물성 예측의 중요성

### 과제 예시

1. **샘플 비교 분석**
   - 서로 다른 암석 타입 3개를 선택하여 물성 비교
   - 공극률이 유사한 샘플들의 다른 물성값 차이 분석

2. **상관관계 도출**
   - 공극률-투수율 그래프에서 추세선 관찰
   - 경험식과 실제 데이터의 차이 분석

3. **응력-변형률 해석**
   - 각 암석의 파괴 메커니즘 추론
   - 취성(brittle)과 연성(ductile) 거동 구분

## 🔧 기술 스택

- **프론트엔드**
  - HTML5 / CSS3
  - Vanilla JavaScript (ES6+)
  - Three.js (3D 렌더링)
  - Chart.js (차트 시각화)

- **데이터 생성**
  - Python 3
  - NumPy (수치 계산)
  - JSON (데이터 저장)

- **물리 모델**
  - Kozeny-Carman equation (공극률-투수율)
  - Wyllie time-average equation (탄성파속도-공극률)
  - 경험적 응력-변형률 모델

## 📁 프로젝트 구조

```
digital_core_simulator/
├── index.html              # 메인 HTML 파일
├── app.js                  # 메인 JavaScript 애플리케이션
├── generate_rock_data.py   # 데이터 생성 스크립트
├── data/                   # 생성된 데이터 디렉토리
│   └── rock_properties_data.json
├── assets/                 # 리소스 파일 (이미지 등)
└── README.md              # 이 파일
```

## 🎯 향후 개선 계획

- [ ] 실제 측정 데이터 업로드 기능
- [ ] 다양한 암석 타입 추가 (편마암, 규암 등)
- [ ] 삼축압축시험 시뮬레이션
- [ ] 공극 네트워크 모델링 (Pore Network Modeling)
- [ ] 유체 흐름 시뮬레이션 (유한요소법)
- [ ] CT 이미지 기반 디지털 코어 생성
- [ ] 머신러닝 기반 물성 예측 모델
- [ ] 다국어 지원 (영어, 한국어)

## 📝 참고문헌

### 암석 물성 데이터
- Carmichael, R. S. (1989). Practical handbook of physical properties of rocks and minerals. CRC press.
- Mavko, G., Mukerji, T., & Dvorkin, J. (2009). The rock physics handbook. Cambridge university press.

### 경험식
- Kozeny, J. (1927). Über kapillare Leitung des Wassers im Boden. Sitzungsber Akad. Wiss., Wien, 136(2a), 271-306.
- Wyllie, M. R. J., Gregory, A. R., & Gardner, G. H. F. (1958). An experimental investigation of factors affecting elastic wave velocities in porous media. Geophysics, 23(3), 459-493.

## 👨‍🏫 수업 정보

- **과목명**: 암석물성계측 실습
- **교육 목표**: 디지털 코어 및 디지털 트윈 개념 이해
- **대상**: 지구과학, 자원공학, 토목공학 전공 학생

## 📄 라이센스

이 프로젝트는 교육 목적으로 자유롭게 사용할 수 있습니다.

## 🙋 문의

프로젝트 관련 문의사항이나 개선 제안은 이슈로 등록해주세요.

---

**Made with ❤️ for Rock Mechanics Education**
