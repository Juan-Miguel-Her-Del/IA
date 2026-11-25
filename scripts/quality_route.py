"""Estado de la ruta y comparación offline; no entrena ni reemplaza modelos."""
import argparse
import json

from utils.config import PROJECT_ROOT
from utils.quality_route import compare_reports, prepare_candidate_config, prepare_experiments, read_json, route_status, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Comprobar datos, dependencias y pesos sin importar torch")
    status.add_argument("--output", help="Guardar un informe nuevo; no sobrescribe")
    plan = commands.add_parser("plan", help="Generar pares control/candidata para GPU de 8 GB; no entrenar")
    plan.add_argument("--output", required=True, help="Carpeta nueva para plan y configuraciones")
    plan.add_argument("--seeds", type=int, nargs="+", default=[72, 73, 74])
    plan.add_argument('--focus', choices=('all', 'material', 'detector'), default='all')
    plan.add_argument('--scenes', default='data/supplemental/scene_materials.zip')
    plan.add_argument('--detector-root', default='data/detection/refine_D2')
    candidate = commands.add_parser("candidate-config", help="Crear una configuración separada para evaluar pesos nuevos")
    candidate.add_argument("--checkpoint", required=True)
    candidate.add_argument("--focus", choices=("material", "detector"), required=True)
    candidate.add_argument("--output", required=True)
    compare = commands.add_parser("compare", help="Comparar dos informes de evaluate_runtime")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--focus", choices=("material", "detector", "pipeline"), default="pipeline")
    compare.add_argument("--policy", default=str(PROJECT_ROOT / "config/quality_route.json"))
    compare.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.command == "status":
            report = route_status()
        elif args.command == "plan":
            report = prepare_experiments(args.output, args.seeds,
                                         focuses=('material', 'detector') if args.focus == 'all' else (args.focus,),
                                         scenes=args.scenes, detector_root=args.detector_root)
            print(f"Plan creado en {args.output}/plan.json. No se ha iniciado entrenamiento.")
            return 0
        elif args.command == "candidate-config":
            report = prepare_candidate_config(args.checkpoint, args.output, args.focus)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        else:
            report = compare_reports(read_json(args.baseline), read_json(args.candidate), read_json(args.policy), args.focus)
        if args.output:
            write_new_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if args.command == "status" or report["offline_pass"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"No se pudo completar: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
