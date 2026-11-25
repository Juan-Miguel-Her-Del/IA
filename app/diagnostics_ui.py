"""Diagnóstico visible y registro voluntario; las lecturas guardadas conservan su fotograma."""
import streamlit as st
from PIL import Image, ImageOps

from app.field_data import FIELD_ROOT, analyze_samples, save_sample, pixel_hash
from app.recycling import LABELS
from app.vision import crop_region


def show_diagnostic(frame, reading, message, service=None):
    with st.expander("Diagnóstico: imagen exacta de la lectura", expanded=True):
        st.caption(message)
        original, crop, resized = st.columns([1.4, 1, 1])
        original.image(frame, caption="Fotograma original RGB, sin anotaciones", width="stretch")
        if reading:
            patch = crop_region(frame, reading.region)
            crop.image(patch, caption="Recorte exacto enviado al preprocesamiento", width="stretch")
            size = reading.image_size
            transform = service.transform if service else None
            preview = transform.prepare_image(Image.fromarray(patch)) if hasattr(transform, "prepare_image") else Image.fromarray(patch).resize((size, size), Image.Resampling.BILINEAR)
            resized.image(preview, caption=f"RGB {size} × {size} antes de normalizar", width=size)
            if service and service.metadata["config"].get("inference_hflip",False):
                resized.image(ImageOps.mirror(preview),caption="Segunda vista: espejo horizontal",width=size)
            x1, y1, x2, y2 = reading.region.box
            ratio = (x2-x1)*(y2-y1)/(frame.shape[0]*frame.shape[1])
            st.write(f"Método: **{reading.region.method}** · caja: {reading.region.box} · "
                     f"recorte: {x2-x1} × {y2-y1} px · {ratio:.1%} del fotograma.")
            st.caption("Comprueba si falta alguna parte del objeto o sobra fondo. Estas proporciones no determinan por sí solas que la localización sea correcta.")
            st.table([{"Categoría": LABELS[label], "Probabilidad softmax": f"{probability:.4%}"}
                      for label, probability in reading.probabilities.items()])
            st.caption("La transformación posterior usa la normalización del checkpoint; no aplica aumentaciones aleatorias.")
        else:
            st.warning("Sin lectura válida. Clasifica una región disponible o cambia a manual / imagen completa si falla la localización.")


def record_panel(frame, reading, message, service, source, key):
    if service is None:
        return
    # Cada foto/recorte exige una confirmación nueva, sin heredar la etiqueta anterior.
    key = key + "_" + pixel_hash(frame)[:16] + "_" + (reading.tensor_sha256[:12] if reading else "sin-region")
    with st.expander("Registrar esta fotografía real"):
        st.caption("Se guardarán la foto, el recorte y la lectura mostrados, separados de los datasets oficiales. "
                   "Si la localización falló, se guarda la foto y el fallo, sin inventar un recorte.")
        with st.form("record_" + key):
            actual = st.selectbox("Categoría verdadera (asignación manual)", [None, *service.classes],
                                  format_func=lambda value: "Seleccionar…" if value is None else LABELS[value], key=key+"_actual")
            confirmed = st.checkbox("Es una foto real nueva y he comprobado la categoría verdadera", key=key+"_confirm")
            if st.form_submit_button("Guardar fotografía y diagnóstico"):
                if actual is None or not confirmed:
                    st.warning("Selecciona la categoría y confirma antes de guardar.")
                else:
                    try:
                        folder = save_sample(frame, reading, actual, service, source, message=message)
                        st.success("Fotografía y diagnóstico guardados.")
                        st.code(str(folder))
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))


def field_report_panel():
    with st.expander("Informe de fotografías reales registradas"):
        st.caption(f"Origen: {FIELD_ROOT}. Máximo 10 fotos por categoría y grupo de checkpoint/método; no se reentrena.")
        if st.button("Generar informe de fotos reales"):
            try:
                report, output = analyze_samples()
                st.success(f"Fotografías disponibles: {report['records_available']}")
                st.code(str(output / "REPORT.md"))
            except (ValueError, OSError, KeyError) as exc:
                st.error(str(exc))
