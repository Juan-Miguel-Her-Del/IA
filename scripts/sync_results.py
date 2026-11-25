"""Importa resultados sincronizados y resume validacion; nunca evalua modelos."""
import argparse
import time

from utils.drive_sync import load_sync_settings, sync_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Revisar periodicamente hasta Ctrl+C; sin servidor.")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.interval < 5 or args.interval > 60:
        parser.error("interval debe estar entre 5 y 60 segundos.")
    settings = load_sync_settings()
    while True:
        try:
            result = sync_results(settings)
            print(f"Ejecuciones locales: {len(result['runs'])}; pendientes: {len(result['pending'])}. Resumen: {result['summary']}", flush=True)
        except (OSError, ValueError) as exc:
            if not args.watch:
                raise
            print(f"Drive no esta listo: {exc}. Se reintentara sin modificar resultados.", flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Observacion detenida; resultados conservados.")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
