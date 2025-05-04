from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

TRAIN_IMG_DIR = DATA_DIR / "Train_Alphabet"
SYNTEHTIC_TEST_IMG_DIR = DATA_DIR / "Test_Alphabet"
REAL_TEST_IMG_DIR = DATA_DIR / "asl-alphabet-test"

TRAIN_LANDMARKS_DIR = DATA_DIR / "Train_Alphabet_Landmarks"
SYNTHETIC_TEST_LANDMARKS_DIR = DATA_DIR / "Test_Alphabet_Landmarks"
REAL_TEST_LANDMARKS_DIR = DATA_DIR / "asl-alphabet-test-landmarks"

REPORTS_DIR = ROOT_DIR / "reports"