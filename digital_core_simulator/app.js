/**
 * Digital Core Simulator - Main Application
 * 디지털 코어 시뮬레이터 메인 애플리케이션
 */

// 전역 변수
let rockData = null;
let currentRockType = null;
let currentSample = null;
let scene, camera, renderer, coreModel, controls;
let stressStrainChart, porosityPermChart, velocityPorosityChart;

// 애플리케이션 초기화
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // 데이터 로드
        await loadRockData();

        // UI 초기화
        initializeRockTypeSelector();
        initializeEventListeners();

        // 3D 렌더러 초기화
        init3DRenderer();

        // 미세구조 캔버스 초기화
        initMicrostructureCanvas();

        // 차트 초기화
        initializeCharts();

        console.log('✓ 디지털 코어 시뮬레이터 초기화 완료');
    } catch (error) {
        console.error('초기화 실패:', error);
        alert('데이터 로드 실패. data/rock_properties_data.json 파일을 확인하세요.');
    }
});

// 데이터 로드
async function loadRockData() {
    const response = await fetch('data/rock_properties_data.json');
    rockData = await response.json();
    console.log('✓ 암석 데이터 로드 완료:', rockData.metadata.rock_types);
}

// 암석 타입 선택기 초기화
function initializeRockTypeSelector() {
    const select = document.getElementById('rockTypeSelect');
    rockData.metadata.rock_types.forEach(rockType => {
        const option = document.createElement('option');
        option.value = rockType;
        option.textContent = rockType;
        select.appendChild(option);
    });
}

// 이벤트 리스너 초기화
function initializeEventListeners() {
    // 암석 타입 선택
    document.getElementById('rockTypeSelect').addEventListener('change', (e) => {
        currentRockType = e.target.value;
        if (currentRockType) {
            updateSampleList();
            updateCorrelationCharts();
        }
    });

    // 슬라이더 이벤트
    document.getElementById('porositySlider').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('porosityValue').textContent = value.toFixed(1) + '%';
        updateVisualization();
    });

    document.getElementById('densitySlider').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('densityValue').textContent = value.toFixed(2) + ' g/cm³';
        updateVisualization();
    });

    document.getElementById('vpSlider').addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        document.getElementById('vpValue').textContent = value + ' m/s';
        updateVisualization();
    });
}

// 샘플 리스트 업데이트
function updateSampleList() {
    const sampleList = document.getElementById('sampleList');
    sampleList.innerHTML = '';

    const samples = rockData.samples[currentRockType];
    samples.forEach((sample, index) => {
        const sampleItem = document.createElement('div');
        sampleItem.className = 'sample-item';
        sampleItem.textContent = `${sample.sample_id} (φ=${sample.porosity.toFixed(1)}%, Vp=${Math.round(sample.vp)} m/s)`;
        sampleItem.addEventListener('click', () => {
            selectSample(sample, index);
            // 모든 샘플 아이템에서 active 제거
            document.querySelectorAll('.sample-item').forEach(item => {
                item.classList.remove('active');
            });
            sampleItem.classList.add('active');
        });
        sampleList.appendChild(sampleItem);
    });

    // 첫 번째 샘플 자동 선택
    if (samples.length > 0) {
        selectSample(samples[0], 0);
        sampleList.firstChild.classList.add('active');
    }
}

// 샘플 선택
function selectSample(sample, index) {
    currentSample = sample;

    // 슬라이더 값 업데이트
    document.getElementById('porositySlider').value = sample.porosity;
    document.getElementById('porosityValue').textContent = sample.porosity.toFixed(1) + '%';

    document.getElementById('densitySlider').value = sample.density;
    document.getElementById('densityValue').textContent = sample.density.toFixed(2) + ' g/cm³';

    document.getElementById('vpSlider').value = sample.vp;
    document.getElementById('vpValue').textContent = Math.round(sample.vp) + ' m/s';

    // 물성값 표시 업데이트
    updatePropertiesDisplay(sample);

    // 시각화 업데이트
    updateVisualization();

    // 응력-변형률 곡선 업데이트
    updateStressStrainChart();
}

// 물성값 표시 업데이트
function updatePropertiesDisplay(sample) {
    document.getElementById('ucsDisplay').textContent = sample.ucs.toFixed(1);
    document.getElementById('tensileDisplay').textContent = sample.tensile_strength.toFixed(1);
    document.getElementById('permDisplay').textContent = sample.permeability.toFixed(3);
    document.getElementById('youngDisplay').textContent = sample.youngs_modulus.toFixed(1);
    document.getElementById('vpvsDisplay').textContent = (sample.vp / sample.vs).toFixed(2);
}

// 3D 렌더러 초기화
function init3DRenderer() {
    const canvas = document.getElementById('renderCanvas');
    const container = canvas.parentElement;

    // Scene 설정
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Camera 설정
    const aspect = container.clientWidth / (container.clientHeight - 60);
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);

    // Renderer 설정
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight - 60);
    renderer.shadowMap.enabled = true;

    // Controls (OrbitControls)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // 조명 추가
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.4);
    pointLight.position.set(-5, 5, -5);
    scene.add(pointLight);

    // 그리드 추가
    const gridHelper = new THREE.GridHelper(10, 10, 0xcccccc, 0xeeeeee);
    scene.add(gridHelper);

    // 초기 코어 모델 생성
    createCoreModel(15, 2.5); // 기본값

    // 애니메이션 루프
    animate();
}

// 코어 모델 생성
function createCoreModel(porosity, density) {
    // 기존 모델 제거
    if (coreModel) {
        scene.remove(coreModel);
    }

    // 코어 그룹 생성
    coreModel = new THREE.Group();

    // 암석 색상 결정
    let color = 0x8B7D6B; // 기본 색상
    if (currentRockType && rockData.rock_type_info[currentRockType]) {
        color = rockData.rock_type_info[currentRockType].color;
    }

    // 실린더 형태의 코어 생성 (직경 2, 높이 4)
    const coreRadius = 1;
    const coreHeight = 4;
    const segments = 32;

    // 메인 코어 바디
    const coreGeometry = new THREE.CylinderGeometry(coreRadius, coreRadius, coreHeight, segments);
    const coreMaterial = new THREE.MeshPhongMaterial({
        color: color,
        shininess: 30,
        transparent: true,
        opacity: 0.9
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    core.castShadow = true;
    core.receiveShadow = true;
    coreModel.add(core);

    // 공극(pore) 시각화 - 공극률에 따라 구멍 개수 조절
    const numPores = Math.floor(porosity * 10); // 공극률에 비례
    const poreGeometry = new THREE.SphereGeometry(0.08, 8, 8);
    const poreMaterial = new THREE.MeshPhongMaterial({
        color: 0x333333,
        transparent: true,
        opacity: 0.6
    });

    for (let i = 0; i < numPores; i++) {
        const pore = new THREE.Mesh(poreGeometry, poreMaterial);

        // 원통 내부에 랜덤하게 배치
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * (coreRadius - 0.15);
        const height = (Math.random() - 0.5) * (coreHeight - 0.3);

        pore.position.set(
            Math.cos(angle) * radius,
            height,
            Math.sin(angle) * radius
        );

        coreModel.add(pore);
    }

    // 코어 외곽 와이어프레임
    const wireframe = new THREE.WireframeGeometry(coreGeometry);
    const line = new THREE.LineSegments(wireframe);
    line.material.color.setHex(0x444444);
    line.material.opacity = 0.3;
    line.material.transparent = true;
    coreModel.add(line);

    // 축 표시
    const axesHelper = new THREE.AxesHelper(3);
    coreModel.add(axesHelper);

    scene.add(coreModel);
}

// 애니메이션 루프
function animate() {
    requestAnimationFrame(animate);

    if (coreModel) {
        // 천천히 회전
        coreModel.rotation.y += 0.005;
    }

    controls.update();
    renderer.render(scene, camera);
}

// 미세구조 캔버스 초기화
function initMicrostructureCanvas() {
    drawMicrostructure();
}

// 미세구조 그리기
function drawMicrostructure() {
    const canvas = document.getElementById('microstructureCanvas');
    const ctx = canvas.getContext('2d');

    // 캔버스 크기 조정
    const container = canvas.parentElement;
    canvas.width = container.clientWidth - 40;
    canvas.height = container.clientHeight - 100;

    const width = canvas.width;
    const height = canvas.height;

    // 배경 (암석 매트릭스)
    let rockColor = '#8B7D6B';
    if (currentRockType && rockData.rock_type_info[currentRockType]) {
        rockColor = rockData.rock_type_info[currentRockType].color;
    }

    ctx.fillStyle = rockColor;
    ctx.fillRect(0, 0, width, height);

    // 공극률 가져오기
    const porosity = parseFloat(document.getElementById('porositySlider').value);

    // 공극 그리기 (랜덤 원으로 표현)
    const numPores = Math.floor(porosity * 15); // 공극률에 비례

    ctx.fillStyle = 'rgba(50, 50, 50, 0.7)';
    for (let i = 0; i < numPores; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const radius = 5 + Math.random() * 15;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // 그레인(grain) 경계 그리기
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 1;

    for (let i = 0; i < 30; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const size = 20 + Math.random() * 40;

        ctx.beginPath();
        for (let j = 0; j < 6; j++) {
            const angle = (Math.PI * 2 / 6) * j + Math.random() * 0.3;
            const px = x + Math.cos(angle) * size;
            const py = y + Math.sin(angle) * size;
            if (j === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.closePath();
        ctx.stroke();
    }

    // 스케일 바 추가
    ctx.fillStyle = 'white';
    ctx.fillRect(width - 120, height - 40, 100, 25);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.strokeRect(width - 120, height - 40, 100, 25);

    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('1 mm', width - 70, height - 22);
}

// 차트 초기화
function initializeCharts() {
    // 응력-변형률 차트
    const ssCtx = document.getElementById('stressStrainChart').getContext('2d');
    stressStrainChart = new Chart(ssCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: '응력-변형률',
                data: [],
                borderColor: '#2a5298',
                backgroundColor: 'rgba(42, 82, 152, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: '변형률 (Strain)' },
                    ticks: { callback: value => value.toFixed(4) }
                },
                y: {
                    title: { display: true, text: '응력 (MPa)' }
                }
            }
        }
    });

    // 공극률-투수율 차트
    const ppCtx = document.getElementById('porosityPermChart').getContext('2d');
    porosityPermChart = new Chart(ppCtx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: {
                    title: { display: true, text: '공극률 (%)' }
                },
                y: {
                    type: 'logarithmic',
                    title: { display: true, text: '투수율 (mD)' }
                }
            }
        }
    });

    // 탄성파속도-공극률 차트
    const vpCtx = document.getElementById('velocityPorosityChart').getContext('2d');
    velocityPorosityChart = new Chart(vpCtx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: {
                    title: { display: true, text: '공극률 (%)' }
                },
                y: {
                    title: { display: true, text: 'P파 속도 (m/s)' }
                }
            }
        }
    });
}

// 응력-변형률 차트 업데이트
function updateStressStrainChart() {
    if (!currentRockType || !stressStrainChart) return;

    const curveData = rockData.stress_strain_curves[currentRockType];
    const data = curveData.strain.map((strain, i) => ({
        x: strain,
        y: curveData.stress[i]
    }));

    stressStrainChart.data.datasets[0].data = data;
    stressStrainChart.data.datasets[0].label = `${currentRockType} (Peak: ${curveData.peak_stress.toFixed(1)} MPa)`;
    stressStrainChart.update();
}

// 상관관계 차트 업데이트
function updateCorrelationCharts() {
    if (!currentRockType) return;

    // 공극률-투수율 차트
    const ppData = rockData.correlations.porosity_permeability[currentRockType];
    const ppPoints = ppData.porosity.map((phi, i) => ({
        x: phi,
        y: ppData.permeability[i]
    }));

    porosityPermChart.data.datasets = [{
        label: currentRockType,
        data: ppPoints,
        borderColor: rockData.rock_type_info[currentRockType].color,
        backgroundColor: rockData.rock_type_info[currentRockType].color + '40',
        borderWidth: 2,
        pointRadius: 0,
        fill: false
    }];
    porosityPermChart.update();

    // 탄성파속도-공극률 차트
    const vpData = rockData.correlations.velocity_porosity[currentRockType];
    const vpPoints = vpData.porosity.map((phi, i) => ({
        x: phi,
        y: vpData.vp[i]
    }));

    velocityPorosityChart.data.datasets = [{
        label: currentRockType,
        data: vpPoints,
        borderColor: rockData.rock_type_info[currentRockType].color,
        backgroundColor: rockData.rock_type_info[currentRockType].color + '40',
        borderWidth: 2,
        pointRadius: 0,
        fill: false
    }];
    velocityPorosityChart.update();
}

// 시각화 업데이트 (슬라이더 조절 시)
function updateVisualization() {
    const porosity = parseFloat(document.getElementById('porositySlider').value);
    const density = parseFloat(document.getElementById('densitySlider').value);
    const vp = parseFloat(document.getElementById('vpSlider').value);

    // 3D 코어 모델 업데이트
    createCoreModel(porosity, density);

    // 미세구조 업데이트
    drawMicrostructure();

    // 물성값 예측 (간단한 경험식 사용)
    if (currentSample) {
        // 공극률 변화에 따른 물성 추정
        const porosityRatio = porosity / currentSample.porosity;

        // 투수율은 공극률의 3제곱에 비례 (Kozeny-Carman)
        const estimatedPerm = currentSample.permeability * Math.pow(porosityRatio, 3);

        // UCS는 공극률에 반비례
        const estimatedUCS = currentSample.ucs * (1 - (porosity - currentSample.porosity) / 100);

        // Vs 추정 (Vp/Vs ratio 유지)
        const vs = vp / (currentSample.vp / currentSample.vs);

        // 표시 업데이트
        document.getElementById('permDisplay').textContent = estimatedPerm.toFixed(3);
        document.getElementById('ucsDisplay').textContent = Math.max(0, estimatedUCS).toFixed(1);
        document.getElementById('vpvsDisplay').textContent = (vp / vs).toFixed(2);
    }
}

// 윈도우 리사이즈 핸들링
window.addEventListener('resize', () => {
    if (camera && renderer) {
        const container = document.getElementById('renderCanvas').parentElement;
        const aspect = container.clientWidth / (container.clientHeight - 60);
        camera.aspect = aspect;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight - 60);
    }

    drawMicrostructure();
});
