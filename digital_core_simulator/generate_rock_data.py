"""
암석 물성 샘플 데이터 생성 스크립트
Rock Properties Sample Data Generator

이 스크립트는 다양한 암석 타입에 대한 물성 데이터를 생성합니다.
- 탄성파속도 (P-wave, S-wave velocity)
- 일축압축강도 (Uniaxial Compressive Strength, UCS)
- 인장강도 (Tensile Strength)
- 공극률 (Porosity)
- 투수율 (Permeability)
- 밀도 (Density)
- 영률 (Young's Modulus)
- 포아송비 (Poisson's Ratio)
"""

import json
import numpy as np
from datetime import datetime

# 암석 타입별 물성값 범위 (실제 문헌값 기반)
ROCK_PROPERTIES = {
    "Granite": {
        "density": (2.60, 2.75),  # g/cm³
        "porosity": (0.5, 3.0),  # %
        "permeability": (1e-6, 1e-4),  # mD (millidarcy)
        "vp": (4500, 6500),  # m/s
        "vs": (2500, 3800),  # m/s
        "ucs": (100, 250),  # MPa
        "tensile_strength": (7, 25),  # MPa
        "youngs_modulus": (30, 75),  # GPa
        "poissons_ratio": (0.15, 0.30),
        "color": "#8B7D6B"
    },
    "Sandstone": {
        "density": (2.20, 2.65),
        "porosity": (5.0, 30.0),
        "permeability": (1, 1000),  # mD
        "vp": (2000, 4500),
        "vs": (1200, 2800),
        "ucs": (20, 170),
        "tensile_strength": (2, 15),
        "youngs_modulus": (5, 40),
        "poissons_ratio": (0.10, 0.35),
        "color": "#C2B280"
    },
    "Limestone": {
        "density": (2.30, 2.75),
        "porosity": (2.0, 25.0),
        "permeability": (0.1, 100),
        "vp": (3500, 6500),
        "vs": (2000, 3500),
        "ucs": (30, 200),
        "tensile_strength": (3, 20),
        "youngs_modulus": (20, 80),
        "poissons_ratio": (0.20, 0.35),
        "color": "#E8DCC4"
    },
    "Shale": {
        "density": (2.00, 2.70),
        "porosity": (3.0, 15.0),
        "permeability": (1e-5, 0.1),
        "vp": (2000, 4000),
        "vs": (1000, 2300),
        "ucs": (10, 100),
        "tensile_strength": (1, 10),
        "youngs_modulus": (5, 35),
        "poissons_ratio": (0.15, 0.35),
        "color": "#696969"
    },
    "Basalt": {
        "density": (2.70, 3.00),
        "porosity": (0.1, 2.0),
        "permeability": (1e-5, 1e-3),
        "vp": (5000, 6800),
        "vs": (2800, 3900),
        "ucs": (150, 350),
        "tensile_strength": (10, 30),
        "youngs_modulus": (50, 100),
        "poissons_ratio": (0.20, 0.35),
        "color": "#3D3D3D"
    },
    "Marble": {
        "density": (2.60, 2.85),
        "porosity": (0.2, 2.0),
        "permeability": (1e-5, 1e-2),
        "vp": (4000, 7000),
        "vs": (2300, 4000),
        "ucs": (50, 200),
        "tensile_strength": (5, 20),
        "youngs_modulus": (30, 90),
        "poissons_ratio": (0.20, 0.35),
        "color": "#F5F5DC"
    }
}


def generate_sample_data(rock_type, num_samples=10):
    """특정 암석 타입에 대한 샘플 데이터 생성"""
    props = ROCK_PROPERTIES[rock_type]
    samples = []

    for i in range(num_samples):
        sample = {
            "sample_id": f"{rock_type}_{i+1:03d}",
            "rock_type": rock_type,
            "density": np.random.uniform(*props["density"]),
            "porosity": np.random.uniform(*props["porosity"]),
            "permeability": np.random.lognormal(
                np.log(np.sqrt(props["permeability"][0] * props["permeability"][1])),
                0.5
            ),
            "vp": np.random.uniform(*props["vp"]),
            "vs": np.random.uniform(*props["vs"]),
            "ucs": np.random.uniform(*props["ucs"]),
            "tensile_strength": np.random.uniform(*props["tensile_strength"]),
            "youngs_modulus": np.random.uniform(*props["youngs_modulus"]),
            "poissons_ratio": np.random.uniform(*props["poissons_ratio"])
        }

        # Vp/Vs ratio가 물리적으로 타당한지 확인 (일반적으로 1.5-2.0)
        vp_vs_ratio = sample["vp"] / sample["vs"]
        if vp_vs_ratio < 1.5 or vp_vs_ratio > 2.2:
            sample["vs"] = sample["vp"] / np.random.uniform(1.6, 2.0)

        samples.append(sample)

    return samples


def generate_stress_strain_curve(rock_type, ucs, youngs_modulus):
    """응력-변형률 곡선 데이터 생성"""
    # 탄성 구간
    elastic_strain = np.linspace(0, ucs / (youngs_modulus * 1000), 50)
    elastic_stress = elastic_strain * youngs_modulus * 1000

    # 소성 구간 (간단한 모델)
    plastic_strain = np.linspace(elastic_strain[-1], elastic_strain[-1] * 1.5, 30)
    plastic_stress = elastic_stress[-1] + (plastic_strain - elastic_strain[-1]) * youngs_modulus * 200

    # 파괴 구간
    failure_strain = np.linspace(plastic_strain[-1], plastic_strain[-1] * 1.2, 20)
    failure_stress = np.linspace(plastic_stress[-1], plastic_stress[-1] * 0.3, 20)

    strain = np.concatenate([elastic_strain, plastic_strain[1:], failure_strain[1:]])
    stress = np.concatenate([elastic_stress, plastic_stress[1:], failure_stress[1:]])

    return {
        "strain": strain.tolist(),
        "stress": stress.tolist(),
        "peak_stress": float(plastic_stress[-1]),
        "peak_strain": float(plastic_strain[-1])
    }


def generate_porosity_permeability_relationship():
    """공극률-투수율 상관관계 데이터 생성 (Kozeny-Carman 관계식 기반)"""
    porosity = np.linspace(0.5, 35, 50)

    relationships = {}
    for rock_type in ROCK_PROPERTIES.keys():
        props = ROCK_PROPERTIES[rock_type]
        k0 = np.mean(props["permeability"])
        phi0 = np.mean(props["porosity"])

        # Kozeny-Carman equation 변형
        permeability = k0 * (porosity / phi0) ** 3 * ((100 - phi0) / (100 - porosity)) ** 2

        relationships[rock_type] = {
            "porosity": porosity.tolist(),
            "permeability": permeability.tolist()
        }

    return relationships


def generate_velocity_porosity_relationship():
    """탄성파속도-공극률 상관관계 데이터 생성"""
    porosity = np.linspace(0, 35, 50)

    relationships = {}
    for rock_type in ROCK_PROPERTIES.keys():
        props = ROCK_PROPERTIES[rock_type]
        vp0 = props["vp"][1]  # 공극률 0일 때의 최대 속도

        # Wyllie time-average equation 기반
        # 1/Vp = phi/Vfluid + (1-phi)/Vmatrix
        vfluid = 1500  # 물의 탄성파속도 (m/s)
        phi_fraction = porosity / 100
        vp = 1 / (phi_fraction / vfluid + (1 - phi_fraction) / vp0)

        relationships[rock_type] = {
            "porosity": porosity.tolist(),
            "vp": vp.tolist()
        }

    return relationships


def main():
    """메인 함수: 모든 데이터를 생성하고 JSON 파일로 저장"""

    # 전체 데이터 구조
    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "description": "암석물성계측 실습용 샘플 데이터",
            "rock_types": list(ROCK_PROPERTIES.keys()),
            "properties": [
                "density", "porosity", "permeability", "vp", "vs",
                "ucs", "tensile_strength", "youngs_modulus", "poissons_ratio"
            ],
            "units": {
                "density": "g/cm³",
                "porosity": "%",
                "permeability": "mD",
                "vp": "m/s",
                "vs": "m/s",
                "ucs": "MPa",
                "tensile_strength": "MPa",
                "youngs_modulus": "GPa",
                "poissons_ratio": "dimensionless"
            }
        },
        "rock_type_info": ROCK_PROPERTIES,
        "samples": {},
        "stress_strain_curves": {},
        "correlations": {
            "porosity_permeability": generate_porosity_permeability_relationship(),
            "velocity_porosity": generate_velocity_porosity_relationship()
        }
    }

    # 각 암석 타입별 샘플 생성
    for rock_type in ROCK_PROPERTIES.keys():
        samples = generate_sample_data(rock_type, num_samples=15)
        data["samples"][rock_type] = samples

        # 대표 샘플로 응력-변형률 곡선 생성
        representative_sample = samples[0]
        data["stress_strain_curves"][rock_type] = generate_stress_strain_curve(
            rock_type,
            representative_sample["ucs"],
            representative_sample["youngs_modulus"]
        )

    # JSON 파일로 저장
    output_file = "data/rock_properties_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ 암석 물성 데이터 생성 완료: {output_file}")
    print(f"✓ 총 {len(ROCK_PROPERTIES)} 종류의 암석 타입")
    print(f"✓ 각 암석당 15개 샘플 생성")
    print(f"✓ 응력-변형률 곡선 포함")
    print(f"✓ 공극률-투수율, 탄성파속도-공극률 상관관계 포함")

    # 통계 출력
    print("\n=== 암석 타입별 물성값 범위 ===")
    for rock_type, props in ROCK_PROPERTIES.items():
        print(f"\n{rock_type}:")
        print(f"  공극률: {props['porosity'][0]:.1f}-{props['porosity'][1]:.1f}%")
        print(f"  P파속도: {props['vp'][0]}-{props['vp'][1]} m/s")
        print(f"  일축압축강도: {props['ucs'][0]}-{props['ucs'][1]} MPa")


if __name__ == "__main__":
    main()
