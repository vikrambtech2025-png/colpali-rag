"""Check which pytorch CUDA indices host torch 2.13.0 win_amd64 cp311 wheels."""
import re
import urllib.request

VERSION = "2.13.0"
PY = "cp311"
for cu in ["cu126", "cu128", "cu130", "cu129", "cu124"]:
    url = f"https://download.pytorch.org/whl/{cu}/torch/"
    try:
        html = urllib.request.urlopen(url, timeout=30).read().decode()
        wheels = re.findall(rf"torch-{VERSION}[^\"<>]*{PY}-[^\"<>]*win_amd64\.whl", html)
        print(cu, "->", wheels[:2] if wheels else "NONE", f"({len(wheels)} total)")
    except Exception as exc:
        print(cu, "ERR", exc)