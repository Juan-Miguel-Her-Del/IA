"""Reglas locales para residuos comunes en Colombia. Fuentes: docs/RECYCLING_KNOWLEDGE.md."""
LABELS = {"plastico": "Plástico", "papel": "Papel", "carton": "Cartón",
          "vidrio": "Vidrio", "metal": "Metal", "organicos": "Orgánicos"}

TIPS = {
    "plastico": ("Vacía el envase y retira los restos de producto.",
                 "Déjalo limpio y seco; consulta qué tipos de plástico recibe tu reciclador."),
    "papel": ("Conserva el papel limpio y seco; evita mezclarlo con comida o líquidos.",
              "Separa papel higiénico, servilletas usadas y papeles con grasa: no van con el papel aprovechable."),
    "carton": ("Vacía y aplana las cajas para facilitar su almacenamiento.",
               "Separa las partes limpias y secas de las zonas con grasa o comida."),
    "vidrio": ("Vacía botellas y frascos; entrégalos limpios y secos.",
               "Evita romperlos. Bombillos, espejos y cerámica requieren otra gestión; consulta su entrega local."),
    "metal": ("Vacía las latas y envases metálicos y déjalos limpios y secos.",
              "No manipules bordes cortantes ni perfores aerosoles; consulta su manejo específico."),
    "organicos": ("Separa restos de alimentos de envases, bolsas y otros materiales.",
                  "Usa la recolección orgánica o el compostaje disponible; sigue las reglas de materiales aceptados."),
}

CONDITIONS = ("Orientación general", "Limpio y seco", "Con restos o humedad", "Residuo especial o peligroso")


def guidance(material, condition="Orientación general"):
    if material not in LABELS or condition not in CONDITIONS:
        raise ValueError("Categoría o condición desconocida.")
    organic = material == "organicos"
    result = {"label": LABELS[material], "group": "Orgánico" if organic else "Inorgánico (grupo operativo)",
              "destination": "Verde · orgánicos aprovechables" if organic else "Blanco · aprovechables limpios y secos",
              "advice": "Separa según el esquema local de aprovechamiento de orgánicos." if organic else
                        "Entrega limpio y seco en la corriente blanca, si lo acepta tu reciclador local.",
              "tips": TIPS[material]}
    if condition == "Con restos o humedad" and not organic:
        result["destination"] = "Revisar antes de separar"
        result["advice"] = "Retira los restos y deja secar si puedes recuperarlo. Consulta su aceptación local."
        if material in ("papel", "carton"):
            result["destination"] = "Negro · partes no aprovechables"
            result["advice"] = "Separa las partes con grasa o comida que no puedan recuperarse; conserva aparte las partes limpias y secas."
    if condition == "Residuo especial o peligroso":
        result["destination"] = "Entrega diferenciada"
        result["advice"] = "No lo mezcles con la recolección común. Consulta un punto o programa autorizado para ese residuo."
        result["tips"] = ("Esta CNN no identifica peligrosidad, composición química ni condiciones sanitarias.",
                          "No abras, huelas ni manipules envases de sustancias desconocidas.")
    return result
