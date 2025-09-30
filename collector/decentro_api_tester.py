import argparse
import json
import os
import time
from datetime import datetime

import pandas as pd
import requests


class DecentroAPIAnalyzer:
    def __init__(self):
        self.results = []

    def test_api(self, url, name, timeout=10):
        start = time.time()
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=timeout)
            end = time.time()

            result = {
                "name": name,
                "url": url,
                "time_ms": round((end - start) * 1000, 2),
                "status": response.status_code,
                "success": response.status_code == 200,
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(result)
            print(f"{name}: {result['time_ms']}ms")
            return result

        except Exception as e:
            print(f"{name} failed: {e}")
            self.results.append(
                {
                    "name": name,
                    "url": url,
                    "time_ms": None,
                    "status": None,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            )
            return None

    def collect_comprehensive_data(self, endpoints, sleep_seconds=1, timeout=10):
        print("Collecting decentro performance data.")
        for url, name in endpoints:
            self.test_api(url, name, timeout=timeout)
            time.sleep(sleep_seconds)

    def save_data(self, out_dir="."):
        if not self.results:
            print("No results to save")
            return None

        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.results)
        filename = f"decentro_api_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = os.path.join(out_dir, filename)
        df.to_csv(out_path, index=False)
        print(f" Saved to {out_path}")
        return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decentro API performance collector.")
    parser.add_argument(
        "--endpoints",
        type=str,
        default="",)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,)
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,)
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,)
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",)
    return parser.parse_args()


def load_endpoints(arg_value):
    default_endpoints = [("https://decentro.tech", "Main_Site"),
        ("https://docs.decentro.tech", "Documentation"),
        ("https://docs.decentro.tech/docs/overview", "API_Overview"),
        ("https://docs.decentro.tech/docs/api-basics", "API_Basics"),
        ("https://decentro.tech/about", "About_Page"),
        ("https://decentro.tech/products", "Products_Page"),]

    if not arg_value:
        return default_endpoints
    if os.path.exists(arg_value) and arg_value.lower().endswith(".json"):
        with open(arg_value, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(arg_value)
    normalized = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append((item[0], item[1]))
        elif isinstance(item, dict) and "url" in item and "name" in item:
            normalized.append((item["url"], item["name"]))
        else:
            raise ValueError(
                "Endpoint items must be ['url','name'] or {'url':..., 'name':...}" )
    return normalized


if __name__ == "__main__":
    args = parse_args()
    endpoints = load_endpoints(args.endpoints)
    analyzer = DecentroAPIAnalyzer()

    for _ in range(max(args.samples, 1)):
        analyzer.collect_comprehensive_data(
            endpoints=endpoints, sleep_seconds=args.sleep, timeout=args.timeout)

    analyzer.save_data(out_dir=args.outdir)
    print("Collector run.")