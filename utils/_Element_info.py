"""
Atomic masses & number information

Constants:
    MASS: Dict[element symbol : atomic mass]

    N_MASS: Dict[atomic number: atomic mass]

    ATOMIC_NUMBER: Dict[atomic number : element symbol]

    ATOMIC_SYMBOL: Dict[element symbol : atomic number]

    NONRADIOACTIVE_METALS: The set of non-radioactive metals

    TRANSITION_METALS: The set of transition metals

    TRANSITION_P_METALS: The set of transition metals & p-zone metals

    NOBLE_METALS: The set of noble metals
"""

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _Element_info.py
#  Environment: Python 3.12

MASS = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Po": 209.0,
    "At": 210.0,
    "Rn": 222.0,
}  # element set

N_MASS = {
    1: 1.008,
    2: 4.0026,
    3: 6.94,
    4: 9.0122,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    10: 20.180,
    11: 22.990,
    12: 24.305,
    13: 26.982,
    14: 28.085,
    15: 30.974,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.098,
    20: 40.078,
    21: 44.956,
    22: 47.867,
    23: 50.942,
    24: 51.996,
    25: 54.938,
    26: 55.845,
    27: 58.933,
    28: 58.693,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.630,
    33: 74.922,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.468,
    38: 87.62,
    39: 88.906,
    40: 91.224,
    41: 92.906,
    42: 95.95,
    43: 98.0,
    44: 101.07,
    45: 102.91,
    46: 106.42,
    47: 107.87,
    48: 112.41,
    49: 114.82,
    50: 118.71,
    51: 121.76,
    52: 127.60,
    53: 126.90,
    54: 131.29,
    55: 132.91,
    56: 137.33,
    57: 138.91,
    58: 140.12,
    59: 140.91,
    60: 144.24,
    61: 145.0,
    62: 150.36,
    63: 151.96,
    64: 157.25,
    65: 158.93,
    66: 162.50,
    67: 164.93,
    68: 167.26,
    69: 168.93,
    70: 173.05,
    71: 174.97,
    72: 178.49,
    73: 180.95,
    74: 183.84,
    75: 186.21,
    76: 190.23,
    77: 192.22,
    78: 195.08,
    79: 196.97,
    80: 200.59,
    81: 204.38,
    82: 207.2,
    83: 208.98,
    84: 209.0,
    85: 210.0,
    86: 222.0,
}  # element set

ATOMIC_NUMBER: 'dict(atomic number: element symbol)' = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
}

ATOMIC_SYMBOL: 'dict(element symbol : atomic number)' = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
}

NONRADIOACTIVE_METALS: "Te set of non-radioactive metals" = {
    "Li",  # 锂
    "Be",  # 铍
    "Na",  # 钠
    "Mg",  # 镁
    "Al",  # 铝
    "K",   # 钾
    "Ca",  # 钙
    "Sc",  # 钪
    "Ti",  # 钛
    "V",   # 钒
    "Cr",  # 铬
    "Mn",  # 锰
    "Fe",  # 铁
    "Co",  # 钴
    "Ni",  # 镍
    "Cu",  # 铜
    "Zn",  # 锌
    "Ga",  # 镓
    "Rb",  # 铷
    "Sr",  # 锶
    "Y",   # 钇
    "Zr",  # 锆
    "Nb",  # 铌
    "Mo",  # 钼
    "Ru",  # 钌
    "Rh",  # 铑
    "Pd",  # 钯
    "Ag",  # 银
    "Cd",  # 镉
    "In",  # 铟
    "Sn",  # 锡
    "Cs",  # 铯
    "Ba",  # 钡
    "La",  # 镧
    "Ce",  # 铈
    "Pr",  # 镨
    "Nd",  # 钕
    "Sm",  # 钐
    "Eu",  # 铕
    "Gd",  # 钆
    "Tb",  # 铽
    "Dy",  # 镝
    "Ho",  # 钬
    "Er",  # 铒
    "Tm",  # 铥
    "Yb",  # 镱
    "Lu",  # 镥
    "Hf",  # 铪
    "Ta",  # 钽
    "W",   # 钨
    "Re",  # 铼
    "Os",  # 锇
    "Ir",  # 铱
    "Pt",  # 铂
    "Au",  # 金
    "Hg",  # 汞
    "Tl",  # 铊
    "Pb",  # 铅
    "Bi"   # 铋
}

TRANSITION_METALS: "The set of transition metals" = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}

TRANSITION_P_METALS: "The set of transition metals & p-zone metals" = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Al",
    "Ga",
    "In",
    "Sn",
    "Tl",
    "Pb",
    "Bi"
}

NOBLE_METALS: "The set of noble metals" = {
    "Rh",
    "Ru",
    "Os",
    "Pt",
    "Ir",
    "Pd",
    "Au"
}
