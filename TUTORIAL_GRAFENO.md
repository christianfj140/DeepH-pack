# Tutorial de uso de DeepH-pack con grafeno

Este repositorio implementa DeepH, un flujo para aprender Hamiltonianos DFT
en base localizada y usarlos luego en estructuras nuevas. En la practica hay
tres etapas:

1. `deeph-preprocess`: convierte salidas DFT crudas a la estructura de datos
   de DeepH.
2. `deeph-train`: entrena el modelo con los Hamiltonianos procesados.
3. `deeph-inference`: predice Hamiltonianos para una estructura objetivo y
   calcula propiedades, por ejemplo una banda.

Para grafeno el repositorio ya trae una configuracion base en
`ini/graphene.ini`. Esa configuracion usa `interface = npz`, pensada para el
dataset de grafeno ya procesado publicado por los autores.

## 1. Entorno Python local

En esta copia del repo se creo un entorno virtual en `.venv/` con Python 3.12
y dependencias GPU para la RTX 5090. Para activarlo:

```bash
cd /home/christian/repositorios/DeepH-pack
source .venv/bin/activate
```

Comprobacion rapida:

```bash
python - <<'PY'
import torch, torch_geometric, torch_scatter, e3nn, pymatgen, deeph
print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("e3nn:", e3nn.__version__)
print("deeph:", deeph.__version__)
PY
```

Los comandos que deberian existir al activar el entorno son:

```bash
deeph-preprocess --help
deeph-train --help
deeph-inference --help
deeph-evaluate --help
```

Para recrear este entorno desde cero en esta maquina:

```bash
cd /home/christian/repositorios/DeepH-pack
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-deeph-py312-cu130.txt
python tools/patch_e3nn_torch_load.py
```

Nota importante: la documentacion original recomienda Python 3.9, PyTorch
1.9.1, PyG 1.7.2 y Julia 1.6.6. Como esta maquina solo tenia Python 3.12
disponible y la GPU es una RTX 5090/Blackwell, el entorno creado usa Torch
2.11.0 con CUDA 13.0 y PyG 2.5.3. Importa DeepH, detecta `cuda:0` y carga sus
CLI correctamente. Si vas a reproducir exactamente resultados del paper en un
cluster antiguo conviene comparar tambien con el stack historico Python 3.9.

## 2. Dependencias que no van dentro del venv

El entorno Python no sustituye a los codigos externos:

- Para crear un dataset desde DFT necesitas uno de los paquetes soportados:
  OpenMX, ABACUS, FHI-aims o SIESTA.
- Para preprocesar salidas OpenMX/FHI-aims se usan scripts Julia del repo.
  Instala Julia y los paquetes documentados si vas a usar esa ruta.
- Para inferencia de bandas puedes evitar Julia en sistemas pequenos usando
  `eigen_solver = dense_py`. Para estructuras grandes normalmente se usa
  `sparse_jl`, que requiere Julia.

## 3. Ruta rapida: entrenar con el dataset de grafeno publicado

Esta es la forma mas directa de usar grafeno porque evita correr DFT y
preprocesado.

En esta copia del repositorio el dataset de Zenodo ya esta preparado en:

```text
/home/christian/repositorios/DeepH-pack/work/data/graphene_dataset/processed
```

La configuracion local lista para entrenar esta en:

```text
/home/christian/repositorios/DeepH-pack/work/config/graphene_train_local.ini
```

Para lanzar el entrenamiento con esa configuracion:

```bash
cd /home/christian/repositorios/DeepH-pack
source .venv/bin/activate
deeph-train --config ini/graphene.ini work/config/graphene_train_local.ini
```

1. Descarga `graphene_dataset.zip` del registro de datos indicado en el README
   de DeepH, Zenodo record `6555484`.
2. Descomprime el dataset en una carpeta local de trabajo, por ejemplo:

```bash
mkdir -p work/data work/config work/graph_data work/result
unzip graphene_dataset.zip -d work/data/
```

3. Copia la configuracion de grafeno:

```bash
cp ini/graphene.ini work/config/graphene_train.ini
```

4. Edita las rutas iniciales de `work/config/graphene_train.ini`:

```ini
[basic]
graph_dir = /home/christian/repositorios/DeepH-pack/work/graph_data/graphene
save_dir = /home/christian/repositorios/DeepH-pack/work/result/graphene
raw_dir = /home/christian/repositorios/DeepH-pack/work/data/graphene_dataset
dataset_name = graphene
interface = npz
disable_cuda = False
device = cuda:0
```

`ini/graphene.ini` ya trae el bloque `orbital` para carbono-carbono, con
numero atomico `6 6` y 13 orbitales por atomo segun el dataset/configuracion
de los autores. No lo cambies salvo que cambies tambien la base DFT.

5. Para un test rapido de instalacion, baja temporalmente las epocas:

```ini
[train]
epochs = 1
```

Eso solo verifica que el flujo arranca. Para entrenar de verdad, restaura un
numero alto de epocas, por ejemplo las `5000` de `ini/graphene.ini`.

6. Ejecuta entrenamiento:

```bash
source .venv/bin/activate
deeph-train --config work/config/graphene_train.ini
```

El resultado se guarda en `save_dir`. Si `save_to_time_folder = True`, DeepH
creara una subcarpeta con marca temporal. Dentro deberian aparecer archivos
como `best_model.pt`, `best_state_dict.pkl`, logs y datos de TensorBoard.

Para ver las curvas de entrenamiento:

```bash
tensorboard --logdir work/result/graphene
```

## 4. Ruta completa: generar tu propio dataset de grafeno desde DFT

Usa esta ruta si quieres entrenar con tus propios calculos de grafeno,
superceldas deformadas, defectos, distintas distancias, etc.

### 4.1 Calculos DFT

Prepara varias estructuras pequenas de grafeno que representen el entorno
quimico de la estructura objetivo. Cada estructura debe estar en una carpeta
separada.

Con OpenMX, el README indica que el input debe incluir:

```text
System.Name   openmx
HS.fileout    On
```

Despues de cada calculo OpenMX, concatena la salida de texto al binario:

```bash
cat openmx.out >> openmx.scfout
```

La carpeta cruda puede quedar asi:

```text
work/raw/graphene_openmx/
  sample_001/openmx.scfout
  sample_002/openmx.scfout
  sample_003/openmx.scfout
```

### 4.2 Preprocesado

Crea un archivo `work/config/graphene_preprocess.ini`:

```ini
[basic]
raw_dir = /home/christian/repositorios/DeepH-pack/work/raw/graphene_openmx
processed_dir = /home/christian/repositorios/DeepH-pack/work/data/graphene_processed
target = hamiltonian
interface = openmx
multiprocessing = 0
local_coordinate = True
get_S = False

[interpreter]
julia_interpreter = julia

[graph]
radius = -1.0
create_from_DFT = True
r2_rand = False

[magnetic_moment]
parse_magnetic_moment = False
magnetic_element = ["Cr", "Mn", "Fe", "Co", "Ni"]
```

Ejecuta:

```bash
source .venv/bin/activate
deeph-preprocess --config work/config/graphene_preprocess.ini
```

La carpeta `processed_dir` es la que luego se usa como `raw_dir` en
`deeph-train`. Si usas ABACUS o SIESTA, cambia `interface` y sigue la
estructura que describen `docs/source/dataset/dataset.rst` y
`deeph/preprocess/preprocess_default.ini`.

### 4.3 Orbitales

La clave `orbital` dice que elementos de matriz Hamiltoniana se predicen. Para
grafeno con la configuracion incluida, el repositorio ya trae todo preparado.

Si cambias la base, genera una cadena nueva con:

```bash
python tools/get_all_orbital_str.py
```

Para carbono, el numero atomico es `6`. El numero de orbitales debe coincidir
con la base localizada de tus calculos DFT.

## 5. Inferencia y banda de grafeno

Despues de entrenar, calcula la matriz de solapamiento de la estructura
objetivo con el mismo codigo DFT y la misma base usada en el entrenamiento.
Para OpenMX se usa la version modificada "overlap only"; para ABACUS se usa
`calculation get_S`.

Copia el ejemplo de banda:

```bash
cp deeph/inference/band_config.json work/config/graphene_band.json
```

Ese JSON ya contiene una ruta hexagonal `Gamma-M-K-Gamma`. Ajusta al menos:

```json
{
  "fermi_level": -3.82373,
  "num_band": 50
}
```

El `fermi_level` debe venir de tu calculo DFT o de la referencia que estes
comparando.

Crea `work/config/graphene_inference.ini`:

```ini
[basic]
work_dir = /home/christian/repositorios/DeepH-pack/work/inference/graphene
OLP_dir = /home/christian/repositorios/DeepH-pack/work/overlap/graphene_target
interface = openmx
trained_model_dir = /home/christian/repositorios/DeepH-pack/work/result/graphene/TU_CARPETA_DE_MODELO
task = [1, 2, 3, 4, 5]
sparse_calc_config = /home/christian/repositorios/DeepH-pack/work/config/graphene_band.json
eigen_solver = dense_py
disable_cuda = False
device = cuda:0
huge_structure = False
restore_blocks_py = True
gen_rc_idx = False
gen_rc_by_idx =
with_grad = False

[interpreter]
julia_interpreter = julia
python_interpreter = /home/christian/repositorios/DeepH-pack/.venv/bin/python

[graph]
radius = -1.0
create_from_DFT = True
```

Ejecuta:

```bash
source .venv/bin/activate
deeph-inference --config work/config/graphene_inference.ini
```

Para estructuras grandes cambia `eigen_solver = sparse_jl`, instala Julia y
sus paquetes, y usa `huge_structure = True`.

## 6. Archivos principales del repo para grafeno

- `ini/graphene.ini`: configuracion de entrenamiento para el dataset publicado.
- `deeph/default.ini`: valores por defecto de entrenamiento.
- `deeph/preprocess/preprocess_default.ini`: valores por defecto de preproceso.
- `deeph/inference/inference_default.ini`: valores por defecto de inferencia.
- `deeph/inference/band_config.json`: ejemplo de configuracion de banda.
- `tools/get_all_orbital_str.py`: generador interactivo de la clave `orbital`.

## 7. Problemas frecuentes

- Si `deeph-train` no encuentra datos, revisa que `raw_dir` apunte al dataset
  procesado, no al ZIP ni a la carpeta padre equivocada.
- Si hay errores de orbitales o dimensiones, la base DFT no coincide con el
  bloque `orbital`.
- Si falla inferencia en `task = 1`, faltan archivos de solapamiento en
  `OLP_dir` o el `interface` no coincide.
- Si falla `sparse_jl`, prueba primero `eigen_solver = dense_py` en una
  estructura pequena para separar problemas Python de problemas Julia.
- Si aparece `propagate() got an unexpected keyword argument 'distance'`, es
  una incompatibilidad entre DeepH y PyG moderno. En este checkout ya esta
  parcheada la linea `propagate_type` de `deeph/model.py`.
- Entrenar en CPU funciona para pruebas, pero para un entrenamiento real de
  grafeno conviene usar `cuda:0` y muchas epocas.
