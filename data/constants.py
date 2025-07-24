# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz

import os


UCR_SETS = [
    'ACSF1',  # 0
    'Adiac',  # 1
    'AllGestureWiimoteX',  # 2
    'AllGestureWiimoteY',  # 3
    'AllGestureWiimoteZ',  # 4
    'ArrowHead',  # 5
    'BME',  # 6
    'Beef',  # 7
    'BeetleFly',  # 8
    'BirdChicken',  # 9
    'CBF',  # 10
    'Car',  # 11
    'Chinatown',  # 12
    'ChlorineConcentration',  # 13
    'CinCECGTorso',  # 14
    'Coffee',  # 15
    'Computers',  # 16
    'CricketX',  # 17
    'CricketY',  # 18
    'CricketZ',  # 19
    'Crop',  # 20
    'DiatomSizeReduction',  # 21
    'DistalPhalanxOutlineAgeGroup',  # 22
    'DistalPhalanxOutlineCorrect',  # 23
    'DistalPhalanxTW',  # 24
    'DodgerLoopDay',  # 25
    'DodgerLoopGame',  # 26
    'DodgerLoopWeekend',  # 27
    'ECG200',  # 28
    'ECG5000',  # 29
    'ECGFiveDays',  # 30
    'EOGHorizontalSignal',  # 31
    'EOGVerticalSignal',  # 32
    'Earthquakes',  # 33
    'ElectricDevices',  # 34
    'EthanolLevel',  # 35
    'FaceAll',  # 36
    'FaceFour',  # 37
    'FacesUCR',  # 38
    'FiftyWords',  # 39
    'Fish',  # 40
    'FordA',  # 41
    'FordB',  # 42
    'FreezerRegularTrain',  # 43
    'FreezerSmallTrain',  # 44
    'Fungi',  # 45
    'GestureMidAirD1',  # 46
    'GestureMidAirD2',  # 47
    'GestureMidAirD3',  # 48
    'GesturePebbleZ1',  # 49
    'GesturePebbleZ2',  # 50
    'GunPoint',  # 51
    'GunPointAgeSpan',  # 52
    'GunPointMaleVersusFemale',  # 53
    'GunPointOldVersusYoung',  # 54
    'Ham',  # 55
    'HandOutlines',  # 56
    'Haptics',  # 57
    'Herring',  # 58
    'HouseTwenty',  # 59
    'InlineSkate',  # 60
    'InsectEPGRegularTrain',  # 61
    'InsectEPGSmallTrain',  # 62
    'InsectWingbeatSound',  # 63
    'ItalyPowerDemand',  # 64
    'LargeKitchenAppliances',  # 65
    'Lightning2',  # 66
    'Lightning7',  # 67
    'Mallat',  # 68
    'Meat',  # 69
    'MedicalImages',  # 70
    'MelbournePedestrian',  # 71
    'MiddlePhalanxOutlineAgeGroup',  # 72
    'MiddlePhalanxOutlineCorrect',  # 73
    'MiddlePhalanxTW',  # 74
    'MixedShapesRegularTrain',  # 75
    'MixedShapesSmallTrain',  # 76
    'MoteStrain',  # 77
    'NonInvasiveFetalECGThorax1',  # 78
    'NonInvasiveFetalECGThorax2',  # 79
    'OSULeaf',  # 80
    'OliveOil',  # 81
    'PLAID',  # 82
    'PhalangesOutlinesCorrect',  # 83
    'Phoneme',  # 84
    'PickupGestureWiimoteZ',  # 85
    'PigAirwayPressure',  # 86
    'PigArtPressure',  # 87
    'PigCVP',  # 88
    'Plane',  # 89
    'PowerCons',  # 90
    'ProximalPhalanxOutlineAgeGroup',  # 91
    'ProximalPhalanxOutlineCorrect',  # 92
    'ProximalPhalanxTW',  # 93
    'RefrigerationDevices',  # 94
    'Rock',  # 95
    'ScreenType',  # 96
    'SemgHandGenderCh2',  # 97
    'SemgHandMovementCh2',  # 98
    'SemgHandSubjectCh2',  # 99
    'ShakeGestureWiimoteZ',  # 100
    'ShapeletSim',  # 101
    'ShapesAll',  # 102
    'SmallKitchenAppliances',  # 103
    'SmoothSubspace',  # 104
    'SonyAIBORobotSurface1',  # 105
    'SonyAIBORobotSurface2',  # 106
    'StarLightCurves',  # 107
    'Strawberry',  # 108
    'SwedishLeaf',  # 109
    'Symbols',  # 110
    'SyntheticControl',  # 111
    'ToeSegmentation1',  # 112
    'ToeSegmentation2',  # 113
    'Trace',  # 114
    'TwoLeadECG',  # 115
    'TwoPatterns',  # 116
    'UMD',  # 117
    'UWaveGestureLibraryAll',  # 118
    'UWaveGestureLibraryX',  # 119
    'UWaveGestureLibraryY',  # 120
    'UWaveGestureLibraryZ',  # 121
    'Wafer',  # 122
    'Wine',  # 123
    'WordSynonyms',  # 124
    'Worms',  # 125
    'WormsTwoClass',  # 126
    'Yoga'  # 127
    ]

UCR_PREFIX = [
    'ACSF1',
    'Adiac',
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'ArrowHead',
    'BME',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'CBF',
    'Car',
    'Chinatown',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'Crop',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
    'ECG200',
    'ECG5000',
    'ECGFiveDays',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'Earthquakes',
    'ElectricDevices',
    'EthanolLevel',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPoint',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'HouseTwenty',
    'InlineSkate',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OSULeaf',
    'OliveOil',
    'PLAID',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'Plane',
    'PowerCons',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    'Rock',
    'ScreenType',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SmoothSubspace',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    'UMD',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga'
    ]


UEA_SETS = ['/UEA/ArticularyWordRecognition',  # 0
            '/UEA/AtrialFibrillation',  #1
            '/UEA/BasicMotions',  #2
            '/UEA/CharacterTrajectories',  #3
            '/UEA/Cricket',  # 4
            '/UEA/DuckDuckGeese',  # 5
            '/UEA/EigenWorms',  # 6
            '/UEA/Epilepsy',  # 7
            '/UEA/EthanolConcentration',  # 8
            '/UEA/ERing',  # 9
            '/UEA/FaceDetection',  # 10
            '/UEA/FingerMovements',  # 11
            '/UEA/HandMovementDirection',  # 12
            '/UEA/Handwriting',  # 13
            '/UEA/Heartbeat',  # 14
            '/UEA/InsectWingbeat',  # 15
            '/UEA/JapaneseVowels',  # 16
            '/UEA/Libras',  # 17
            '/UEA/LSST',  # 18
            '/UEA/MotorImagery',  # 19
            '/UEA/NATOPS',  # 20
            '/UEA/PenDigits',  # 21
            '/UEA/PEMS-SF',  # 22
            '/UEA/PhonemeSpectra',  # 23
            '/UEA/RacketSports',  # 24
            '/UEA/SelfRegulationSCP1',  # 25
            '/UEA/SelfRegulationSCP2',  # 26
            '/UEA/SpokenArabicDigits',  # 27
            '/UEA/StandWalkJump',  # 28
            '/UEA/UWaveGestureLibrary']  # 29#


UEA_PREFIX = [
    'ArticularyWordRecognition',
    'AtrialFibrillation',
    'BasicMotions',
    'CharacterTrajectories',
    'Cricket',
    'DuckDuckGeese',
    'EigenWorms',
    'Epilepsy',
    'EthanolConcentration',
    'ERing',
    'FaceDetection',
    'FingerMovements',
    'HandMovementDirection',
    'Handwriting',
    'Heartbeat',
    'InsectWingbeat',
    'JapaneseVowels',
    'Libras',
    'LSST',
    'MotorImagery',
    'NATOPS',
    'PenDigits',
    'PEMS-SF',
    'PhonemeSpectra',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'StandWalkJump',
    'UWaveGestureLibrary']


UCR_NEW_PREFIX = [
'ACSF1',
'AconityMINIPrinterLarge_eq',
'AconityMINIPrinterSmall_eq',
'Adiac',
'AllGestureWiimoteX_eq',
'AllGestureWiimoteY_eq',
'AllGestureWiimoteZ_eq',
'ArrowHead',
'AsphaltObstaclesUni_eq',
'AsphaltPavementTypeUni_eq',
'AsphaltRegularityUni_eq',
'BME',
'Beef',
'BeetleFly',
'BirdChicken',
'CBF',
'Car',
'Chinatown',
'ChlorineConcentration',
'CinCECGTorso',
'Coffee',
'Colposcopy',
'Computers',
'Covid3Month_disc',
'CricketX',
'CricketY',
'CricketZ',
'Crop',
'DiatomSizeReduction',
'DistalPhalanxOutlineAgeGroup',
'DistalPhalanxOutlineCorrect',
'DistalPhalanxTW',
'DodgerLoopDay_nmv',
'DodgerLoopGame_nmv',
'DodgerLoopWeekend_nmv',
'ECG200',
'ECG5000',
'ECGFiveDays',
'EOGHorizontalSignal',
'EOGVerticalSignal',
'Earthquakes',
'ElectricDeviceDetection',
'ElectricDevices',
'EthanolLevel',
'FaceAll',
'FaceFour',
'FacesUCR',
'FiftyWords',
'Fish',
'FloodModeling1_disc',
'FloodModeling2_disc',
'FloodModeling3_disc',
'FordA',
'FordB',
'FreezerRegularTrain',
'FreezerSmallTrain',
'GestureMidAirD1_eq',
'GestureMidAirD2_eq',
'GestureMidAirD3_eq',
'GesturePebbleZ1_eq',
'GesturePebbleZ2_eq',
'GunPoint',
'GunPointAgeSpan',
'GunPointMaleVersusFemale',
'GunPointOldVersusYoung',
'Ham',
'HandOutlines',
'Haptics',
'Herring',
'HouseTwenty',
'InlineSkate',
'InsectEPGRegularTrain',
'InsectEPGSmallTrain',
'InsectWingbeatSound',
'ItalyPowerDemand',
'KeplerLightCurves',
'LargeKitchenAppliances',
'Lightning2',
'Lightning7',
'Mallat',
'Meat',
'MedicalImages',
'MelbournePedestrian_nmv',
'MiddlePhalanxOutlineAgeGroup',
'MiddlePhalanxOutlineCorrect',
'MiddlePhalanxTW',
'MixedShapesRegularTrain',
'MixedShapesSmallTrain',
'MoteStrain',
'NonInvasiveFetalECGThorax1',
'NonInvasiveFetalECGThorax2',
'OSULeaf',
'OliveOil',
'PLAID_eq',
'PhalangesOutlinesCorrect',
'PhoneHeartbeatSound',
'Phoneme',
'PickupGestureWiimoteZ_eq',
'PigAirwayPressure',
'PigArtPressure',
'PigCVP',
'Plane',
'PowerCons',
'ProximalPhalanxOutlineAgeGroup',
'ProximalPhalanxOutlineCorrect',
'ProximalPhalanxTW',
'RefrigerationDevices',
'Rock',
'ScreenType',
'SemgHandGenderCh2',
'SemgHandMovementCh2',
'SemgHandSubjectCh2',
'ShakeGestureWiimoteZ_eq',
'ShapeletSim',
'ShapesAll',
'SharePriceIncrease',
'SmallKitchenAppliances',
'SmoothSubspace',
'SonyAIBORobotSurface1',
'SonyAIBORobotSurface2',
'StarLightCurves',
'Strawberry',
'SwedishLeaf',
'Symbols',
'SyntheticControl',
'ToeSegmentation1',
'ToeSegmentation2',
'Tools',
'Trace',
'TwoLeadECG',
'TwoPatterns',
'UMD',
'UWaveGestureLibraryAll',
'UWaveGestureLibraryX',
'UWaveGestureLibraryY',
'UWaveGestureLibraryZ',
'Wafer',
'Wine',
'WordSynonyms',
'Worms',
'WormsTwoClass',
'Yoga'
]
