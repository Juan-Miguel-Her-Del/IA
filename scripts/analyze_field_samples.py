"""Resume únicamente fotografías reales registradas; no carga modelos ni datasets oficiales."""
import argparse

from app.field_data import FIELD_ROOT, analyze_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default=str(FIELD_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-per-class", type=int, default=10)
    args = parser.parse_args()
    report, path = analyze_samples(args.samples_dir, args.output_dir, args.max_per_class)
    print(f"Fotografías reales disponibles: {report['records_available']}. Informe: {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
