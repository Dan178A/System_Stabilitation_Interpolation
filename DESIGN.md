# DESIGN.md — Estabilizador ML

Sistema visual de la interfaz web. Cualquier cambio en `frontend/index.html`
se calibra contra este documento.

## 1. Clasificación

**APP UI**, no landing page. El usuario llega con una tarea (estabilizar un clip
y leer las métricas), no a ser convencido. Todas las reglas de abajo derivan de
eso. La versión anterior aplicaba reglas de landing page a una herramienta, que
era la causa raíz del aspecto de plantilla generada.

## 2. Estructura

```
┌──────────────────────────────────────────────────────────────┐
│ BARRA  marca · descriptor          estado motor · [Método]   │ 56px
├────────────────────┬─────────────────────────────────────────┤
│ RAIL  380px        │ ESCENARIO  1fr                          │
│                    │                                         │
│ 01 FUENTE          │  título + chips de la ejecución         │
│    dropzone        │  ┌───────────────────────────────────┐  │
│                    │  │ VIEWPORT                          │  │
│ 02 MÉTODO          │  │  vacío → en proceso → comparación  │  │
│    radiogroup ×4   │  └───────────────────────────────────┘  │
│                    │  transporte                             │
│ 03 PARÁMETROS      │                                         │
│    sliders         │  TABLA calidad (antes/después/Δ)        │
│    ▸ motor         │  STRIP métricas del motor               │
│                    │                                         │
│ [ Estabilizar ]    │                                         │
└────────────────────┴─────────────────────────────────────────┘
```

Reglas de la estructura:

- Dos columnas con **scroll independiente**. Los controles nunca salen de
  pantalla mientras se mira el resultado: se puede cambiar de método y volver a
  ejecutar sin desplazarse.
- El viewport es **una sola pieza de pantalla que muta de estado**
  (vacío → progreso → comparación). No aparecen ni desaparecen tarjetas: el
  arco emocional ocurre en el mismo rectángulo.
- La jerarquía del escenario es: viewport > tabla de calidad > métricas del
  motor. Si solo cupieran tres cosas, serían esas tres.
- El rail se ordena por el orden real de la tarea, numerado en mono (01/02/03).
- Sin hero. La procedencia académica vive en el panel «Método» y no compite con
  la herramienta por la primera pantalla.

## 3. Tokens

| Token | Valor | Uso |
|---|---|---|
| `--bg-0` | `#0E1113` | lienzo del escenario |
| `--bg-1` | `#14181B` | paneles (rail, barra, tablas) |
| `--bg-2` | `#1A1F23` | controles, cabeceras de tabla |
| `--bg-3` | `#22282D` | hover |
| `--bg-stage` | `#08090A` | fondo del video |
| `--line` | `#262C31` | regla hairline estructural |
| `--line-2` | `#333B42` | borde de control |
| `--ink` | `#E6E8EA` | texto principal |
| `--ink-2` | `#9BA1A8` | texto secundario |
| `--ink-3` | `#868E95` | notas y etiquetas (mínimo AA: 6.07:1) |
| `--accent` | `#4E8FD4` | **único acento cromático** — selección, foco, estabilizado |
| `--accent-deep` | `#2E5B8C` | bordes del acento |
| `--accent-wash` | `#4E8FD41F` | fondo de selección |
| `--warm` | `#C8894A` | semántica «original / fuente» |
| `--ok` `--err` | `#5FA37E` `#C9635C` | delta positivo / error |

Radios: **solo 3px y 5px**. Nada redondeado más allá de eso salvo el punto de
estado y el knob de división, que son círculos por función.

Sombras: **ninguna**. La elevación se comunica con superficie y regla, no con
sombra. El diseño debe verse premium con cero sombras decorativas — es la prueba
de fuego, y aquí se aprueba por construcción.

## 4. Tipografía

- **IBM Plex Sans** — 400 / 500 / 600. Interfaz, prosa, encabezados.
- **IBM Plex Mono** — 400 / 500. **Todas las cifras**, etiquetas de sección,
  chips, metadatos, códigos de estado.

Dos familias, una superfamilia: coherencia sin monotonía. Procedencia de diseño
industrial real (Bold Monday para IBM), que es exactamente el registro de un
instrumento de medición.

Toda cifra lleva `font-variant-numeric: tabular-nums`. Sin excepciones: las
columnas de la tabla de métricas deben alinearse dígito a dígito.

Prohibidos: Inter, Roboto, Arial, `system-ui`, `-apple-system`, Space Grotesk.

Tamaños: cuerpo 14px, secundario 12.5px, notas 11.5–12px, etiquetas mono
10–10.5px con `letter-spacing: .1em`. Ninguna cadena de texto de lectura baja de
11.5px, y ninguna baja de 4.5:1 de contraste.

## 5. Voz

Lenguaje de utilidad: orientación, estado, acción. Nunca ánimo ni aspiración.

- Sí: «Escenario vacío», «motor listo», «División 50% · arrastra la línea o usa ← →»
- No: «Estabiliza tu video con inteligencia artificial», «Visión por computadora ·
  Trabajo de Investigación LUZ» como badge decorativo.

Los encabezados de sección dicen qué es el área o qué puede hacer el usuario.
Cada mensaje de error nombra qué falló **y qué hacer después**.

## 6. Prohibiciones (lo que causaba el aspecto de IA)

Ninguna de estas puede reintroducirse:

1. Gradientes de cualquier tipo — en texto, botones, barras, marcas o bordes.
2. Manchas radiales de color en el fondo.
3. Glow: `box-shadow` que use un color de acento.
4. Glassmorphism / `backdrop-filter` decorativo.
5. Hero centrado, `text-align:center` como recurso de página.
6. Pill badges con punto pulsante.
7. Radios grandes y uniformes (16–24px) en todo.
8. `border-left: 3px solid <acento>` como decoración de tarjeta.
9. Emojis como iconografía.
10. Tarjetas apiladas donde corresponde una tabla o un layout.
11. Rejilla simétrica de tres columnas con icono en círculo de color.

Las tarjetas se ganan su existencia: aquí solo existe una lista de métodos
(la fila **es** la interacción). Las métricas son una **tabla**, porque son
datos comparables columna contra columna.

## 7. Estados

| Zona | Vacío | Cargando | Error | Éxito |
|---|---|---|---|---|
| Fuente | dropzone con formatos aceptados | — | nombre del archivo + causa + formatos válidos | fila con archivo, tamaño y «Cambiar» |
| Método | siempre poblado (fallback local si el motor no responde) | — | punto rojo «motor sin responder», métodos locales | punto verde «motor listo» |
| Ejecutar | deshabilitado + nota que explica por qué | «Procesando…» | alerta `role="alert"` sobre el botón | «Cambia el método y vuelve a ejecutar» |
| Viewport | esquema del método + qué hacer a continuación | % grande, barra 2px, stepper de 7 etapas | vuelve a vacío, error en el rail | comparación con división |
| Métricas | ocultas | ocultas | ocultas | tabla + strip |

El estado vacío del viewport **no es un placeholder**: dibuja el propio método
(trayectoria con temblor sobre malla → trayectoria suavizada) y dice el
siguiente paso. Es la primera explicación del producto.

## 8. Accesibilidad

- Métodos: `role="radiogroup"` / `role="radio"`, navegación con flechas,
  `tabindex` móvil.
- División de comparación: `role="slider"` con `aria-valuenow` / `aria-valuetext`,
  ← → (±2), Shift+← → (±10), Home / End. **Ya no es solo arrastrable.**
- Área de agarre del knob: 44×44px. Golpe extendido de la línea: ±22px.
- Sliders: `aria-valuetext` en lenguaje natural («media resolución», no «0.5»).
- Progreso: `aria-live="polite"`. Errores: `role="alert"`.
- Panel Método: `role="dialog"`, `aria-modal`, cierre con Esc, foco devuelto al
  disparador.
- `:focus-visible` global de 2px en acento con offset.
- El color nunca es el único indicador: los deltas llevan flecha y signo, el
  estado del motor lleva texto, los métodos llevan `aria-checked`.
- `prefers-reduced-motion` desactiva animaciones, conteos y scroll suave.

## 9. Responsive

| Ancho | Layout |
|---|---|
| ≥1100px | dos columnas, 380px + 1fr, scroll independiente, página sin scroll |
| 760–1099px | una columna: rail arriba, escenario abajo; barra de acción **sticky** al pie |
| <640px | strip de métricas apilado, tabla condensada (se ocultan las notas de fila), descriptor de la barra oculto |

«Apilado en móvil» no es una decisión: en cada corte cambia qué elemento manda.
La barra de acción sticky existe porque en una columna el botón de ejecutar
quedaría a una pantalla de distancia de los parámetros.

## 10. Correcciones funcionales incluidas

- **Etiquetas invertidas en la comparación.** La capa `after-wrap` recortaba
  desde la izquierda, así que el video *estabilizado* se mostraba en la mitad
  izquierda mientras la etiqueta de esa mitad decía «Original». Ahora la capa
  estabilizada usa `clip-path: inset(0 0 0 var(--split))`: izquierda = original,
  derecha = estabilizada, coincidiendo con las etiquetas.
- Eliminado el ajuste de ancho en píxeles del video superpuesto y su handler de
  `resize`: `clip-path` no lo necesita.
- El viewport toma la relación de aspecto real del video (`--ar`) en
  `loadedmetadata`, con `object-fit: contain`, para no recortar el encuadre que
  el usuario está evaluando.
- Resincronización de los dos videos cuando derivan más de 150 ms.
- El contrato con el backend no cambió: `/api/methods`, `/api/stabilize`,
  `/api/progress/{id}` (SSE), `/api/result/{id}`, `/api/video/{id}/{kind}`, y los
  mismos campos de `FormData`.
