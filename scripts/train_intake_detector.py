"""Local D2 continuation on the current six-material dataset; no dataset writes."""
import argparse
import json
from pathlib import Path
from training.intake_detection import train_detector

if __name__ == '__main__':
    # El checkpoint padre fija los pesos propios desde los que continúa D2.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train_intake_detector.json')
    parser.add_argument('--parent', default='models/runtime/detector_D2.pth')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train_detector(json.loads(Path(args.config).read_text(encoding='utf-8')), args.parent, args.output, args.smoke)
