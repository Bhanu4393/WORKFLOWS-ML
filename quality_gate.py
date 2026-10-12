import json
import sys

MIN_ACCURACY = 0.60

with open("metrics.json") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]
print(f"Model accuracy: {accuracy}  (minimum required: {MIN_ACCURACY})")

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
