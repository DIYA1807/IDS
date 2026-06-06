"""
Extract and display attack samples from NSL-KDD dataset for testing
"""
import pandas as pd
import requests
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Column names for NSL-KDD dataset
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

CAT_COLS = ["protocol_type", "service", "flag"]
TEST_PATH = r"C:\Users\asd\OneDrive\Desktop\PROJECT\IDS\nsl-kdd\KDDTest+.txt"
TRAIN_PATH = r"C:\Users\asd\OneDrive\Desktop\PROJECT\IDS\nsl-kdd\KDDTrain+.txt"

# Create label encoders for categorical features
def create_encoders():
    """Create and fit label encoders on full dataset"""
    train_df = pd.read_csv(TRAIN_PATH, names=COL_NAMES)
    test_df = pd.read_csv(TEST_PATH, names=COL_NAMES)
    
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        encoders[col] = le
    
    return encoders

ENCODERS = create_encoders()

def load_test_data():
    """Load test dataset"""
    df = pd.read_csv(TEST_PATH, names=COL_NAMES)
    return df

def get_attack_samples():
    """Get samples of each attack type"""
    df = load_test_data()
    
    print("=" * 80)
    print("ATTACK TYPES IN DATASET")
    print("=" * 80)
    
    # Show unique attack types
    attack_counts = df['label'].value_counts()
    print("\nAttack Type Distribution:")
    print(attack_counts)
    
    # Get one sample of each attack type
    print("\n" + "=" * 80)
    print("SAMPLE ATTACK DATA")
    print("=" * 80)
    
    for attack_type in df['label'].unique():
        sample = df[df['label'] == attack_type].iloc[0]
        
        print(f"\n\n{'─' * 80}")
        print(f"ATTACK: {attack_type.upper()}")
        print(f"{'─' * 80}")
        print(sample[:-1].to_string())  # Exclude difficulty column
        
        return sample  # Return first sample for testing

def test_via_api(sample_row):
    """Send sample to API for testing"""
    print("\n\n" + "=" * 80)
    print("TESTING VIA API")
    print("=" * 80)
    
    # Encode categorical features
    protocol_type_encoded = float(ENCODERS["protocol_type"].transform([sample_row["protocol_type"]])[0])
    service_encoded = float(ENCODERS["service"].transform([sample_row["service"]])[0])
    flag_encoded = float(ENCODERS["flag"].transform([sample_row["flag"]])[0])
    
    # Build request payload
    payload = {
        "duration": float(sample_row.get("duration", 0)),
        "protocol_type": protocol_type_encoded,
        "service": service_encoded,
        "flag": flag_encoded,
        "src_bytes": float(sample_row.get("src_bytes", 0)),
        "dst_bytes": float(sample_row.get("dst_bytes", 0)),
        "num_failed_logins": float(sample_row.get("num_failed_logins", 0)),
        "logged_in": float(sample_row.get("logged_in", 0)),
        "root_shell": float(sample_row.get("root_shell", 0)),
        "su_attempted": float(sample_row.get("su_attempted", 0)),
        "num_shells": float(sample_row.get("num_shells", 0)),
        "num_access_files": float(sample_row.get("num_access_files", 0)),
        "count": float(sample_row.get("count", 0)),
        "srv_count": float(sample_row.get("srv_count", 0)),
        "serror_rate": float(sample_row.get("serror_rate", 0)),
        "srv_serror_rate": float(sample_row.get("srv_serror_rate", 0)),
        "rerror_rate": float(sample_row.get("rerror_rate", 0)),
        "same_srv_rate": float(sample_row.get("same_srv_rate", 0)),
        "diff_srv_rate": float(sample_row.get("diff_srv_rate", 0)),
        "dst_host_count": float(sample_row.get("dst_host_count", 0)),
        "dst_host_srv_count": float(sample_row.get("dst_host_srv_count", 0)),
        "dst_host_same_srv_rate": float(sample_row.get("dst_host_same_srv_rate", 0)),
        "dst_host_diff_srv_rate": float(sample_row.get("dst_host_diff_srv_rate", 0)),
        "dst_host_same_src_port_rate": float(sample_row.get("dst_host_same_src_port_rate", 0)),
        "dst_host_srv_diff_host_rate": float(sample_row.get("dst_host_srv_diff_host_rate", 0)),
        "dst_host_serror_rate": float(sample_row.get("dst_host_serror_rate", 0)),
        "dst_host_srv_serror_rate": float(sample_row.get("dst_host_srv_serror_rate", 0)),
        "dst_host_rerror_rate": float(sample_row.get("dst_host_rerror_rate", 0)),
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("\n✅ PREDICTION RESULT:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n⚠️  Cannot connect to API. Make sure the server is running:")
        print(f"   cd C:\\Users\\asd\\OneDrive\\Desktop\\PROJECT\\IDS")
        print(f"   python main.py")
        print(f"\nError: {e}")

if __name__ == "__main__":
    print("\n🔍 NSL-KDD ATTACK SAMPLE VIEWER\n")
    
    # Load and show attack samples
    sample = get_attack_samples()
    
    # Test via API if available
    test_via_api(sample)
