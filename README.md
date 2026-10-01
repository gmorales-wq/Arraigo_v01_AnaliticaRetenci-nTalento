# Riesgo de rotación voluntaria · App Streamlit

Aplicación web para detectar de forma temprana el riesgo de baja voluntaria en un horizonte de 6–12 meses y priorizar acciones de retención.

Combina dos enfoques que se procesan en local, sin IA generativa ni servicios externos:

- un **score de reglas ponderadas**, explicable y disponible desde el primer día;
- un **modelo de aprendizaje automático** (regresión logística con scikit-learn) que aprende del histórico de bajas y estima una probabilidad de baja.

> **Herramienta de apoyo a la decisión.** El score sirve para orientar conversaciones entre responsable y persona y para priorizar recursos. No debe ser la base única de ninguna decisión sobre una persona.

---

## 1. Estructura del proyecto

```
rotacion-app/
├── app.py                  # Interfaz Streamlit (solo presentación)
├── config.py               # Esquema, factores, pesos, umbrales y límites por defecto
├── data.py                 # Lectura de CSV/XLSX, saneado y validación
├── create_mock_data.py     # Generador de datos ficticios (módulo y script)
├── scoring.py              # Score, tramos, contribuciones y validación frente a bajas
├── recommendations.py      # Reglas del plan de acción y exportación segura a CSV
├── model.py                # Modelo predictivo: regresión logística, validación cruzada, explicación
├── fairness.py             # Auditoría de equidad agregada con supresión de grupos pequeños
├── requirements.txt        # Dependencias de ejecución (versiones fijadas)
├── requirements-dev.txt    # + pytest
├── pytest.ini
├── .gitignore              # Excluye CSV/XLSX y secretos del repositorio
├── .streamlit/
│   └── config.toml         # Oculta trazas al usuario, límite de subida, tema
└── tests/
    ├── test_data.py        # Validación de esquema, PII, rangos, formatos
    ├── test_core.py        # Scoring, valores ausentes, plan, CSV, equidad
    ├── test_model.py       # Entrenamiento, explicación y exclusión de atributos sensibles
    └── test_app.py         # Prueba de humo de la interfaz (streamlit.testing)
```

## 2. Instalación y ejecución local

Python **3.13** es la versión probada. El mínimo es **3.12**, porque numpy 2.5.x lo exige.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Para ejecutar los tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Para generar un CSV ficticio y probar la carga de archivos:

```bash
python create_mock_data.py --n 500 --seed 42 --out datos_ficticios.csv
python create_mock_data.py --sensibles --out datos_con_edad_genero.csv
```

## 3. Despliegue en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub. El `.gitignore` excluye `*.csv`, `*.xlsx` y `secrets.toml`; no lo desactives.
2. En Community Cloud, crea la app apuntando a `app.py`.
3. En *Advanced settings*, elige Python 3.13 o 3.12. Comprueba qué versiones ofrece la plataforma en ese momento, porque cambian con el tiempo.
4. La app no necesita secretos: no hay claves de API.

**Usa Community Cloud solo para demostraciones con datos ficticios.** Antes de cargar datos reales de empleados hay que evaluar varias cosas:

- dónde se alojan y procesan los datos (posibles transferencias internacionales);
- quién puede acceder a la app (las apps pueden ser públicas o privadas);
- el contrato de encargo de tratamiento con el proveedor de alojamiento.

Para uso real, lo prudente es desplegar en infraestructura propia o contratada con esas garantías.

## 4. Datos

### Esquema

| Campo | Tipo | Rango / valores | Obligatorio |
|---|---|---|---|
| ID_Empleado | texto | seudónimo, p. ej. `EMP-0001` | Sí |
| Departamento | categórica | libre | Sí |
| Puesto | categórica | libre | Sí |
| Nivel | ordinal | Junior, Medio, Senior, Mánager | Sí |
| Antigüedad_meses | entero | 0–480 | Sí |
| Evaluación_Desempeño_2024 | decimal | 1–5 | Sí |
| Evaluación_Desempeño_2025 | decimal | 1–5 | Sí |
| Potencial | ordinal | Bajo, Medio, Alto | Sí |
| Nivel_Competencias | entero | 1–5 | Sí |
| Horas_Formación_12m | entero | 0–120 | Sí |
| Meses_desde_última_promoción | entero | 0–480 | Sí |
| Compa_Ratio | decimal | 0,7–1,3 | Sí |
| Horas_Extra_Mensuales_Media | decimal | 0–200 | Sí |
| Cambios_Responsable_24m | entero | 0–4 | Sí |
| Puntuación_Clima | decimal | 1–5 | Sí |
| Días_Ausencia_12m | entero | 0–365 (recuento, sin causa) | Sí |
| Modalidad_Trabajo | categórica | Presencial, Híbrido, Remoto | Sí |
| Baja_Voluntaria | sí/no | Sí/No, 1/0, True/False | No (solo histórico) |
| Edad | entero | 16–80 | No (solo auditoría) |
| Género | categórica | libre | No (solo auditoría) |

Desde la barra lateral se puede descargar una plantilla CSV.

### Qué hace la carga de archivos

- **Formatos:** acepta CSV (UTF-8 o Latin-1; separador `;`, `,`, tabulador o `|`; coma decimal admitida) y XLSX.
- **Límites:** el tamaño máximo es configurable, sin superar los 50 MB del servidor. El máximo de filas es 100.000.
- **Nombres de columna:** tolera tildes, mayúsculas y espacios (`Evaluacion Desempeno 2025` se reconoce). También acepta algunos sinónimos (`Sexo` → `Género`).
- **Identificadores directos:** elimina las columnas cuyo nombre sugiere nombre, apellidos, email, DNI/NIE, teléfono, dirección, fecha de nacimiento, etc., y avisa.
- **Causas de ausencia:** rechaza las columnas con causa o diagnóstico de ausencias (`causa`, `motivo`, `diagnóstico`…), porque podrían contener datos de salud.
- **Minimización:** descarta cualquier otra columna ajena al esquema (p. ej. estado civil).
- **Errores bloqueantes:** faltan columnas obligatorias, hay IDs vacíos o duplicados, o hay IDs con aspecto de email o DNI/NIE. En ese caso se pide seudonimizar el archivo antes de cargarlo.
- **Advertencias:** valores no numéricos, fuera de rango o categorías no admitidas. Esas celdas se tratan como vacías y el informe indica columna y filas (numeradas como en Excel).

## 5. Cómo se calcula el score

1. Cada factor convierte una variable en una **intensidad de riesgo de 0 a 1** mediante una rampa lineal. La intensidad vale 0 en el umbral «sin riesgo», 1 en el umbral «riesgo máximo» y se recorta fuera de ese intervalo.
2. El **score (0–100)** es la media ponderada de las intensidades de los factores que tienen dato:

   `score = 100 · Σ (peso_i · intensidad_i) / Σ peso_i` (solo factores con dato)

3. La **contribución** de cada factor es `100 · peso_i · intensidad_i / Σ peso_i`. Las contribuciones suman exactamente el score, así que cada resultado es explicable.
4. **Valores ausentes:** si falta un dato, ese factor se excluye del cálculo de esa persona; no se interpreta como «sin riesgo». La columna *Factores con dato* (p. ej. `8/9`) muestra con cuánta información se ha calculado.

### Factores y valores por defecto

> **Supuestos, no evidencia.** Estos pesos y umbrales se han elegido como supuestos razonables para una demostración. No se han derivado de datos empíricos ni de estudios concretos. Revísalos con tu equipo y contrástalos con tu histórico de bajas.

| Factor | Variable | Sin riesgo | Riesgo máximo | Peso |
|---|---|---|---|---|
| Salario por debajo de banda | Compa_Ratio | 1,00 | 0,80 | 18 |
| Clima bajo | Puntuación_Clima | 3,5 | 2,0 | 18 |
| Estancamiento de carrera | Meses_desde_última_promoción | 24 | 60 | 15 |
| Desempeño reciente bajo | Evaluación 2025 | 3,5 | 2,0 | 10 |
| Caída de desempeño | Eval. 2025 − Eval. 2024 | 0,0 | −1,0 | 8 |
| Sobrecarga | Horas extra mensuales | 10 | 30 | 10 |
| Antigüedad en ventana crítica | Antigüedad_meses | 60 | 24 | 8 |
| Inestabilidad de liderazgo | Cambios de responsable 24m | 1 | 3 | 8 |
| Baja inversión en formación | Horas de formación 12m | 20 | 0 | 5 |
| Ausencias elevadas | Días de ausencia 12m | 8 | 20 | **0** |

**El factor de ausencias está desactivado por defecto.** Las ausencias pueden reflejar salud, discapacidad o responsabilidades de cuidado. Usarlas para puntuar riesgo puede penalizar indirectamente a esos colectivos. Actívalo solo después de valorarlo en la EIPD.

### Tramos, talento clave y prioridades

- **Tramos:** bajo (< 25), medio (25–34,9) y alto (≥ 35).
  - Como el score es una media de muchos factores, rara vez supera 50.
  - Estos umbrales se fijaron mirando la distribución de los datos ficticios (en torno a un 25 % en tramo medio o alto y un 5–10 % en alto).
  - **Con datos reales hay que recalibrarlos.**
- **Talento clave:** evaluación 2025 ≥ 4 o potencial alto.
- **Prioridades del plan de acción:**

  | Prioridad | Tramo | Talento clave |
  |---|---|---|
  | P1 | Alto | Sí |
  | P2 | Alto | No |
  | P3 | Medio | Sí |
  | P4 | Medio | No |
  | P5 | Bajo | Seguimiento ordinario |

- **Recomendaciones:** se generan a partir de los hasta 3 factores que más aportan (al menos 4 puntos cada uno). Las reglas están en el diccionario `RULES` de `recommendations.py`, que se puede editar sin tocar la lógica.

### Validación frente a bajas reales

Si el archivo incluye `Baja_Voluntaria`, el dashboard muestra:

- el **AUC**, calculado con el estadístico de Mann-Whitney;
- la **precisión** y la **cobertura** del tramo alto;
- la **matriz de confusión**.

Lee estas métricas con cautela:

- **Circularidad con datos ficticios.** Las bajas ficticias se generaron con los mismos factores que usa el score. Un buen AUC con esos datos no demuestra nada.
- **Fuga temporal.** Las variables deben estar medidas *antes* de la salida. Si el archivo mezcla datos actuales con bajas pasadas, las métricas se inflan o se sesgan.
- **Muestras pequeñas.** Se avisa por debajo de 200 personas o de 30 bajas. Es un umbral heurístico, no un criterio estadístico formal.

### Modelo de aprendizaje automático

La pestaña **Modelo predictivo** entrena una regresión logística (`model.py`).

- **Variables:** los factores activos del score de reglas, con sus valores brutos. Edad y Género nunca se usan; hay un test que lo comprueba.
- **Preparación:** imputación por la mediana, estandarización y regularización L2 (C = 1).
- **Datos de entrenamiento:** filas con `Baja_Voluntaria` conocida. Se necesitan al menos 30 bajas y 30 permanencias.
- **Plantilla actual:** las filas con `Baja_Voluntaria` vacía no se usan para entrenar; el modelo las puntúa.
- **Evaluación:** AUC con validación cruzada estratificada de hasta 5 particiones, comparada con el AUC del score de reglas en las mismas personas.
- **Explicación:** coeficientes estandarizados (efecto de subir una desviación típica) y, para cada persona, el aporte de cada variable al logit.
- **Probabilidades mostradas:** proceden del modelo final, entrenado con todo el histórico. Para quien ya tiene resultado son retrospectivas.
- **Equidad:** la auditoría muestra también la probabilidad media del modelo por grupo.

Con los datos ficticios, el modelo obtiene un AUC algo menor que las reglas. Es esperable: las bajas ficticias se generaron con umbrales muy parecidos a los del score. No permite concluir qué enfoque es mejor con datos reales.

## 6. Auditoría de equidad

La pestaña aparece solo si los datos contienen `Edad` o `Género`. Estas columnas **nunca intervienen en el score**; hay un test que lo comprueba.

- **Edad en tramos:** la edad se muestra agrupada (< 30, 30–44, 45–54, ≥ 55), nunca de forma individual.
- **Comparación entre grupos:** se comparan el score medio, el score mediano, el % en riesgo alto y el ratio frente al grupo con menor tasa.
- **Supresión de grupos pequeños:** se ocultan los grupos con menos de *k* personas (por defecto 10, configurable).
  - Si solo hay un grupo suprimido, se suprime también el siguiente más pequeño (supresión complementaria).
  - El motivo: el dashboard muestra totales, y con un solo grupo oculto sus cifras podrían deducirse por diferencia.
- **Posibles proxies:**
  - para la edad, la correlación de Spearman entre la edad y cada factor;
  - para el género, la diferencia de intensidad media de cada factor entre grupos.

Con los datos ficticios, la antigüedad aparece como proxy de la edad. Es deliberado: ilustra que excluir una variable no elimina el sesgo.

La regla de los cuatro quintos se cita solo como referencia orientativa. Procede del contexto estadounidense de procesos de selección: no es un criterio legal en España ni está pensada para este uso.

## 7. Protección de datos y cumplimiento

> No es asesoramiento jurídico. Contrasta estos puntos con tu DPO o asesoría legal antes de usar datos reales.

- **Seudonimización, no anonimización.** Sustituir nombres por IDs no anonimiza los datos: siguen siendo datos personales según el RGPD. Edad y género no son identificadores directos, pero combinados con departamento o puesto pueden permitir reidentificar a alguien en equipos pequeños.
- **Elaboración de perfiles (art. 22 RGPD).** El scoring de riesgo de rotación es elaboración de perfiles. La app no toma decisiones automatizadas con efectos sobre las personas, y no debe usarse para ello: cualquier decisión requiere intervención humana con criterio propio.
- **EIPD.** Antes de usar datos reales se recomienda una evaluación de impacto relativa a la protección de datos (art. 35 RGPD).
- **Representación de los trabajadores (España).** Según el art. 64.4.d del Estatuto de los Trabajadores, la representación legal de las personas trabajadoras tiene derecho a ser informada de los parámetros, reglas e instrucciones de los algoritmos o sistemas de IA que afectan a decisiones sobre condiciones de trabajo, acceso y mantenimiento del empleo, incluida la elaboración de perfiles.
- **Reglamento europeo de IA.** Esto es una interpretación, no una conclusión jurídica.
  - Un sistema basado solo en reglas definidas por personas, que ejecuta operaciones sin inferir sus propios criterios a partir de datos, probablemente queda fuera de la definición de «sistema de IA» del art. 3.1. La propia norma (considerando 12) y las directrices de la Comisión sobre esa definición apuntan en ese sentido.
  - **El modelo de aprendizaje automático sí infiere sus criterios a partir de datos**, así que con él la app pasa a ser, muy probablemente, un sistema de IA. Sus usos en empleo (supervisar o evaluar el comportamiento de las personas trabajadoras, o influir en decisiones sobre la relación laboral) encajan en el Anexo III, punto 4, es decir, en la categoría de **alto riesgo**. Eso implica, entre otras, obligaciones de gestión de riesgos, gobernanza de datos, documentación técnica, supervisión humana y transparencia.
  - **Calendario:** la fecha de aplicación de las obligaciones de alto riesgo estaba en revisión mediante la propuesta «Digital Omnibus» de la Comisión. Verifica el calendario vigente antes de planificar.
- **Caché.**
  - Los resultados intermedios se guardan con `@st.cache_data` en la memoria del servidor, con un máximo de 30 minutos (`CACHE_TTL_SECONDS` en `app.py`).
  - La caché es compartida por el proceso del servidor. Las entradas se indexan por el contenido exacto de los datos, de modo que otra sesión no puede recuperarlas sin aportar esos mismos datos.
  - El botón *Borrar datos de la sesión y caché* vacía la caché y el estado de la sesión. Al vaciar la caché se borran también las entradas de las demás sesiones del mismo servidor.
- **Registros.** Los errores no controlados se registran solo con el tipo de excepción y la línea de código, sin el mensaje, porque este podría contener valores de los datos.
- **Errores visibles para el usuario.**
  - `.streamlit/config.toml` fija `showErrorDetails = "none"`.
  - El antiguo valor `false` está obsoleto y equivale a `"stacktrace"`, que sí muestra trazas.
- **Exportación.**
  - El CSV del plan neutraliza la inyección de fórmulas: las celdas que empiezan por `=`, `+`, `-` o `@` se prefijan con un apóstrofo.
  - Contiene datos seudonimizados de personas, así que debe guardarse con acceso restringido.

## 8. Limitaciones

- **Supuestos sin validar.** Los pesos, umbrales y tramos son supuestos; ninguno está validado con datos reales.
- **El modelo hereda el pasado.** El modelo aprende de decisiones y circunstancias pasadas, incluidos sus posibles sesgos. Con poco histórico sus coeficientes son inestables.
- **Asociación no es causa.** El score refleja asociaciones supuestas, no causas. Un factor alto es un motivo para conversar, no un diagnóstico.
- **Calidad de los datos de entrada.** El resultado depende de cómo se mida cada variable en la organización: escalas de evaluación, diseño de la encuesta de clima o definición de banda salarial.
- **KPI de bajas.** «Bajas voluntarias en el conjunto» es una proporción sobre los datos cargados. Solo equivale a una tasa anual si el archivo cubre exactamente 12 meses.
- **Rango de Compa_Ratio.** El rango admitido (0,7–1,3) puede no cubrir todos los casos reales. Los valores fuera de rango se tratan como vacíos y se avisa.
- **Supresión simple.** La supresión de grupos protege frente a deducciones simples, pero no frente a todos los ataques de reidentificación. Por ejemplo, alguien que combine varios filtros sucesivos podría acotar resultados.
