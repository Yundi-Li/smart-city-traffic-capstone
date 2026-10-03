"""Test script for the Traffic Volume Prediction API."""

import argparse
import requests

parser = argparse.ArgumentParser(description="Test the Traffic Volume Prediction API")
parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
_args, _ = parser.parse_known_args()
BASE_URL = f"http://localhost:{_args.port}"

SAMPLE_REQUESTS = [
    {
        "label": "Weekday morning, clear skies",
        "payload": {
            "hour": 8,
            "day_of_week": 1,
            "is_weekend": 0,
            "temp": 280.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 10,
            "weather_main": "Clear",
        },
    },
    {
        "label": "Weekend evening, rain",
        "payload": {
            "hour": 19,
            "day_of_week": 5,
            "is_weekend": 1,
            "temp": 290.0,
            "rain_1h": 5.0,
            "snow_1h": 0.0,
            "clouds_all": 90,
            "weather_main": "Rain",
        },
    },
    {
        "label": "Weekday noon, cloudy",
        "payload": {
            "hour": 12,
            "day_of_week": 3,
            "is_weekend": 0,
            "temp": 295.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 75,
            "weather_main": "Clouds",
        },
    },
    {
        "label": "Weekday late night, snow",
        "payload": {
            "hour": 2,
            "day_of_week": 0,
            "is_weekend": 0,
            "temp": 260.0,
            "rain_1h": 0.0,
            "snow_1h": 3.0,
            "clouds_all": 100,
            "weather_main": "Snow",
        },
    },
    {
        "label": "Holiday midday, clear",
        "payload": {
            "hour": 12,
            "day_of_week": 0,
            "is_weekend": 0,
            "temp": 270.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 5,
            "weather_main": "Clear",
            "is_holiday": 1,
        },
    },
    {
        "label": "Weekday rush hour, fog",
        "payload": {
            "hour": 17,
            "day_of_week": 4,
            "is_weekend": 0,
            "temp": 275.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 100,
            "weather_main": "Fog",
        },
    },
]


def test_health():
    """Test the /health endpoint."""
    print("=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Status:     {data['status']}")
        print(f"  Model type: {data['model_type']}")
        print(f"  HTTP {resp.status_code} OK")
    except requests.exceptions.ConnectionError:
        print("  ERROR: Could not connect. Is the server running on port 8000?")
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
    print()


def test_predict():
    """Test the /predict endpoint with multiple sample inputs."""
    print("=" * 60)
    print("Testing /predict endpoint")
    print("=" * 60)
    for sample in SAMPLE_REQUESTS:
        print(f"\n  Scenario: {sample['label']}")
        print(f"  Input:    {sample['payload']}")
        try:
            resp = requests.post(
                f"{BASE_URL}/predict",
                json=sample["payload"],
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"  Predicted volume:   {data['predicted_traffic_volume']}")
            print(f"  Congestion level:   {data['congestion_level']}")
            print(f"  HTTP {resp.status_code} OK")
        except requests.exceptions.ConnectionError:
            print("  ERROR: Could not connect. Is the server running on port 8000?")
        except requests.exceptions.HTTPError as e:
            print(f"  ERROR: {e} — {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: {e}")
    print()


def test_status():
    """Test the /status endpoint (monitoring results)."""
    print("=" * 60)
    print("Testing /status endpoint")
    print("=" * 60)
    try:
        resp = requests.get(f"{BASE_URL}/status", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Overall: {data.get('overall_status', 'UNKNOWN')} — {data.get('overall_message', '')}")
        if "prediction_error_drift" in data:
            ped = data["prediction_error_drift"]
            print(f"  Error drift: {ped['status']} (holdout MAE={ped['holdout_mae']}, prod MAE={ped['production_mae']})")
        for fd in data.get("feature_drift", []):
            print(f"  {fd['feature']:20s} {fd['status']} (KS={fd['ks_statistic']}, p={fd['p_value']})")
        print(f"  HTTP {resp.status_code} OK")
    except requests.exceptions.ConnectionError:
        print("  ERROR: Could not connect. Is the server running on port 8000?")
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
    print()


if __name__ == "__main__":
    test_health()
    test_status()
    test_predict()
    print("All tests completed.")
