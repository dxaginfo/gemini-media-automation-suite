import os
import json
from pathlib import Path

# Create config directory if it doesn't exist
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

# Create default config file
config = {
    "standards": ["rule_of_thirds", "golden_ratio", "leading_lines", "balanced_frame"],
    "suggestionThreshold": 0.7,
    "storageSettings": {
        "resultsBucket": "scene-validator-results",
        "tempBucket": "scene-validator-temp"
    }
}

# Write config to file
with open(config_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("SceneValidator setup complete")
print("Make sure to set the following environment variables:")
print("  - GOOGLE_APPLICATION_CREDENTIALS")
print("  - GEMINI_API_KEY")
print("  - STORAGE_BUCKET (optional)")
