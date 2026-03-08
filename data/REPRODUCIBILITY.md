# Reprodutibilidade do Pipeline de Dados

Documento de reprodutibilidade para a construção do dataset unificado de deteção de elementos GUI, combinando os datasets Rico (mobile) e WebUI (web) em formato YOLO.

**Dissertação:** Detection of Graphical User Interface Elements Using Deep Learning
**Autor:** Rodrigo Santos Magalhães
**Instituição:** ISEP — Mestrado em Engenharia Informática
**Data:** Março 2026

---

## 1. Visão Geral

O pipeline produz um dataset unificado com **105.130 imagens** e **3.317.974 anotações** em **12 classes** de elementos GUI, combinando:

| Dataset | Origem | Plataforma | Imagens | Anotações |
|---------|--------|------------|---------|-----------|
| **Rico** | CMU / Northwestern (UIST 2017) | Android mobile | 63.160 | 1.075.398 |
| **WebUI** | CMU (CHI 2023 Best Paper) | Web (6 viewports) | 41.970 | 2.242.576 |
| **Combinado** | — | Cross-platform | **105.130** | **3.317.974** |

### 1.1 Taxonomia Unificada (12 classes)

| ID | Classe | Rico (mobile) | WebUI (web) |
|----|--------|--------------|-------------|
| 0 | Button | Text Button | button |
| 1 | Text | Text | heading, StaticText, paragraph, strong, emphasis |
| 2 | Image | Image, Video | img, image |
| 3 | Icon | Icon | — |
| 4 | Input | Input, EditText, Slider, Spinner, Date Picker, Number Stepper | textbox, combobox, searchbox, spinbutton, slider |
| 5 | Link | — | link |
| 6 | Checkbox | Checkbox, CheckedTextView, Radio Button | checkbox, radio |
| 7 | Toggle | Switch, On/Off Switch | switch |
| 8 | Toolbar | Toolbar, Button Bar | toolbar |
| 9 | Navigation | Bottom Navigation, Drawer | navigation, menubar, menu |
| 10 | Modal | Modal | dialog, alertdialog |
| 11 | Tab | Multi-Tab, Tab | tab |

### 1.2 Distribuição Final de Classes

```
Button     :  150.613 ( 4.5%)
Text       : 2.054.330 (61.9%)
Image      :  290.838 ( 8.8%)
Icon       :  178.815 ( 5.4%)
Input      :   30.372 ( 0.9%)
Link       :  535.022 (16.1%)
Checkbox   :   10.146 ( 0.3%)
Toggle     :    2.119 ( 0.1%)
Toolbar    :   35.575 ( 1.1%)
Navigation :   21.613 ( 0.7%)
Modal      :    4.097 ( 0.1%)
Tab        :    4.434 ( 0.1%)
```

> **Nota:** Existe desequilíbrio significativo — Text (62%) domina, enquanto Toggle/Modal/Tab representam ~0.1% cada.

---

## 2. Pré-requisitos

### 2.1 Software

| Requisito | Versão testada | Instalação |
|-----------|---------------|------------|
| Python | 3.10+ | — |
| requests | qualquer | `pip install requests` |
| gdown | 5.x | `pip install gdown` |
| PyYAML | qualquer | `pip install pyyaml` |
| Pillow | qualquer | `pip install Pillow` |
| 7-Zip | 23.x | Instalador oficial ou via NVIDIA App |

### 2.2 Verificação do 7-Zip

O script `download_webui.py` procura automaticamente o 7z nos seguintes caminhos:

```
7z                                              (PATH do sistema)
C:\Program Files\7-Zip\7z.exe                  (instalação padrão)
C:\Program Files (x86)\7-Zip\7z.exe            (instalação 32-bit)
C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe  (bundled NVIDIA)
```

### 2.3 Espaço em Disco

| Componente | Tamanho aproximado |
|------------|-------------------|
| Rico screenshots (unique_uis.tar.gz) | ~6 GB download, ~8 GB extraído |
| Rico semantic annotations | ~150 MB download |
| WebUI 7k-balanced (multi-part zip) | ~15 GB download, ~20 GB extraído |
| Dataset YOLO final (unified/combined) | ~30 GB |
| **Total estimado** | **~60-70 GB** |

### 2.4 Instalação de Dependências

```bash
pip install requests gdown pyyaml Pillow
```

---

## 3. Estrutura dos Ficheiros do Pipeline

```
data/
├── class_mapping.py       # Taxonomia unificada + mapeamentos Rico/WebUI
├── prepare_rico.py        # Download + conversão Rico → YOLO
├── download_webui.py      # Download WebUI via Google Drive (gdown + 7z)
├── prepare_webui.py       # Conversão WebUI → YOLO
├── merge_datasets.py      # Merge Rico + WebUI → dataset combinado + data.yaml
└── REPRODUCIBILITY.md     # Este documento
```

---

## 4. Passo a Passo

### Passo 1 — Download e conversão do Rico

```bash
cd data/

# Download (6 GB + 150 MB) e conversão numa só invocação:
python prepare_rico.py --download --convert
```

**O que faz:**
1. Faz download de `unique_uis.tar.gz` (66K screenshots Android, 1440×2560 px) do Google Cloud Storage
2. Faz download de `semantic_annotations.zip` (anotações com `componentLabel`) do Google Cloud Storage
3. Extrai ambos para `data/rico/`
4. Percorre os JSONs de anotações semânticas em formato árvore (campos `componentLabel` + `bounds` + `children`)
5. Mapeia cada `componentLabel` para a classe unificada via `RICO_TO_UNIFIED`
6. Filtra bounding boxes com área < 100 px
7. Converte para formato YOLO normalizado: `class_id cx cy w h` (valores em [0, 1])
8. Divide em train/val/test (80/10/10) com `seed=42`
9. Grava imagens + labels em `data/unified/rico/{train,val,test}/{images,labels}/`

**Resultado esperado:**
```
Screens with valid annotations: 63.160
Total elements extracted: 1.075.398
Skipped labels (not mapped):
  List Item:   247.236
  Card:         43.629
  ...
train: 50.528 images
val:    6.316 images
test:   6.316 images
```

**Estrutura de saída:**
```
data/unified/rico/
├── train/
│   ├── images/    rico_12345.jpg
│   └── labels/    rico_12345.txt    ← "0 0.345612 0.234123 0.150000 0.040000"
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Passo 2 — Download do WebUI

```bash
python download_webui.py
```

**O que faz:**
1. Usa `gdown` para fazer download da pasta `webui-7k-balanced` do Google Drive (~15 GB, multi-part zip)
2. Deteta automaticamente o executável `7z` (verifica vários caminhos)
3. Extrai o arquivo multi-part `.zip.001` com 7z
4. Move as pastas de crawl para `data/webui_raw/`
5. Limpa os ficheiros temporários de `data/webui_tmp/`

**Resultado esperado:**
```
Found 7z: C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe
Downloading webui-7k-balanced from Google Drive...
Extracting with 7z...
Moved ~7000 crawl folders to: data/webui_raw/
```

**Estrutura dos dados raw:**
```
data/webui_raw/
├── {crawl_id}/
│   ├── default_1280-720-axtree.json.gz     ← accessibility tree (nós flat)
│   ├── default_1280-720-bb.json.gz          ← bounding boxes por DOM node ID
│   ├── default_1280-720-screenshot.webp     ← screenshot
│   ├── default_1920-1080-axtree.json.gz
│   ├── default_1920-1080-bb.json.gz
│   ├── default_1920-1080-screenshot.webp
│   ├── iPad-Pro-axtree.json.gz
│   ├── iPad-Pro-bb.json.gz
│   ├── iPad-Pro-screenshot.webp
│   ├── iPhone-13 Pro-axtree.json.gz
│   ├── iPhone-13 Pro-bb.json.gz
│   ├── iPhone-13 Pro-screenshot.webp
│   └── ...                                   (6 viewports por crawl)
```

### Passo 3 — Conversão do WebUI para YOLO

```bash
python prepare_webui.py --raw --webui_dir ./webui_raw
```

**O que faz:**
1. Percorre os ~7.000 diretórios de crawl
2. Para cada viewport, carrega dois ficheiros gzip:
   - `{device}-axtree.json.gz`: lista flat de nós da accessibility tree (Chrome DevTools Protocol)
   - `{device}-bb.json.gz`: mapa `{backendDOMNodeId → {x, y, width, height}}`
3. Faz **join** entre os nós da axtree e as bounding boxes via o campo `backendDOMNodeId`
4. Mapeia cada `role.value` ARIA para a classe unificada via `WEBUI_TO_UNIFIED`
5. Determina dimensões da imagem a partir do nome do viewport (ex: `default_1920-1080` → 1920×1080)
6. Filtra bounding boxes com área < 100 px
7. Converte para formato YOLO normalizado
8. Divide em train/val/test (80/10/10) com `seed=42`
9. Copia screenshots e grava labels em `data/unified/webui/{train,val,test}/{images,labels}/`

**Viewports e escalas:**

| Device | Resolução | Escala | Pixels reais |
|--------|-----------|--------|-------------|
| default_1280-720 | 1280×720 | 1× | 1280×720 |
| default_1366-768 | 1366×768 | 1× | 1366×768 |
| default_1536-864 | 1536×864 | 1× | 1536×864 |
| default_1920-1080 | 1920×1080 | 1× | 1920×1080 |
| iPad-Pro | 1024×1366 | 2× | 2048×2732 |
| iPhone-13 Pro | 390×844 | 3× | 1170×2532 |

**Resultado esperado:**
```
Found 7000 crawl directories
  Processed 1000/7000 crawl dirs...
  Processed 2000/7000 crawl dirs...
  ...
Valid screens with annotations: 41.970
Total elements extracted: 2.242.576

Skipped roles (not mapped):
  generic:     568.234
  listitem:    198.123
  ...

train: 33.576 images
val:    4.197 images
test:   4.197 images
```

### Passo 4 — Merge dos datasets

```bash
python merge_datasets.py --rico_dir ./unified/rico --webui_dir ./unified/webui
```

**O que faz:**
1. Copia imagens e labels de Rico e WebUI para `data/unified/combined/`
2. Gera `data.yaml` com os caminhos absolutos e as 12 classes
3. Imprime estatísticas completas

**Resultado esperado:**
```
Merging datasets...
  train: 84.104 images copied
  val:   10.513 images copied
  test:  10.513 images copied

Total: 105.130 images, 3.317.974 annotations
```

**Ficheiro `data.yaml` gerado:**
```yaml
path: D:/mestrado/Dissertacao_GUI_Detection/data/unified/combined
train: train/images
val: val/images
test: test/images
nc: 12
names:
  - Button
  - Text
  - Image
  - Icon
  - Input
  - Link
  - Checkbox
  - Toggle
  - Toolbar
  - Navigation
  - Modal
  - Tab
```

---

## 5. Resumo Completo (todos os comandos)

```bash
cd data/

# 1. Rico: download + conversão (~30 min, depende da rede)
python prepare_rico.py --download --convert

# 2. WebUI: download (~1-2h, depende da rede)
python download_webui.py

# 3. WebUI: conversão para YOLO (~10-20 min)
python prepare_webui.py --raw --webui_dir ./webui_raw

# 4. Merge final + data.yaml (~5-10 min)
python merge_datasets.py --rico_dir ./unified/rico --webui_dir ./unified/webui
```

**Tempo total estimado:** ~2-3 horas (maioritariamente download).

---

## 6. Formato dos Dados

### 6.1 Formato YOLO (labels .txt)

Cada ficheiro `.txt` contém uma linha por elemento detetado:

```
class_id center_x center_y width height
```

Todos os valores são normalizados para [0, 1] relativamente à dimensão da imagem.

**Exemplo (`rico_12345.txt`):**
```
0 0.345612 0.234123 0.150000 0.040000
1 0.500000 0.120000 0.800000 0.030000
4 0.500000 0.450000 0.600000 0.050000
```

### 6.2 Estrutura Final

```
data/unified/combined/
├── data.yaml              ← configuração YOLO
├── train/
│   ├── images/            ← 84.104 imagens (.jpg, .webp)
│   └── labels/            ← 84.104 ficheiros .txt
├── val/
│   ├── images/            ← 10.513 imagens
│   └── labels/            ← 10.513 ficheiros .txt
└── test/
    ├── images/            ← 10.513 imagens
    └── labels/            ← 10.513 ficheiros .txt
```

---

## 7. Decisões Técnicas e Justificações

### 7.1 Formato dos dados WebUI

O dataset WebUI utiliza o formato Chrome DevTools Protocol (CDP):
- **`axtree.json.gz`**: lista flat de nós da accessibility tree (NÃO uma árvore recursiva). Cada nó tem `role`, `name`, `backendDOMNodeId`.
- **`bb.json.gz`**: dicionário separado que mapeia `backendDOMNodeId` (como string) → `{x, y, width, height}`.

A extração requer um **join** entre os dois ficheiros via `backendDOMNodeId`. Isto difere do formato Rico que tem tudo num único JSON em árvore.

### 7.2 Mapeamento de classes

- **Roles ARIA não mapeados** (ex: `generic`, `listitem`, `section`, `group`, `banner`, `form`): ignorados por serem contentores estruturais sem significado visual como elemento GUI.
- **`Radio Button` → Checkbox**: agrupados por serem visualmente similares na deteção.
- **`Slider`/`Spinner`/`Date Picker` → Input**: agrupados por serem variantes de entrada de dados.
- **`StaticText`/`paragraph` → Text**: mapeados como texto, consistente com o label Rico "Text".

### 7.3 Filtragem

- **Área mínima = 100 px**: filtra artefactos e elementos invisíveis.
- **Bounds clipped a [0, 1]**: garante que coordenadas YOLO são válidas.
- **Nós `ignored: True`**: filtrados na extração da axtree.

### 7.4 Split determinístico

- Seed fixa (`SEED = 42`) e ratio 80/10/10 garantem reprodutibilidade.
- O split é feito por **screen** (não por anotação) para evitar data leakage.

---

## 8. Fontes dos Dados

### Rico
- **Paper:** Deka et al., "Rico: A Mobile App Dataset for Building Data-Driven Design Applications", UIST 2017
- **URL:** https://interactionmining.org/rico
- **Download direto:** Google Cloud Storage (`crowdstf-rico-uiuc-4540`)
- **Licença:** Creative Commons

### WebUI
- **Paper:** Wu et al., "WebUI: A Dataset for Enhancing Visual UI Understanding with Web Semantics", CHI 2023 (Best Paper)
- **URL:** https://uimodeling.github.io/
- **Download:** Google Drive (pasta `webui-7k-balanced`)
- **Subset utilizado:** `webui-7k-balanced` (7.000 páginas × 6 viewports = ~42.000 screenshots)

---

## 9. Resolução de Problemas Comuns

| Problema | Causa | Solução |
|----------|-------|---------|
| `7z not found` | 7-Zip não instalado ou não no PATH | Instalar 7-Zip; o script verifica caminhos comuns automaticamente |
| `0 elements extracted` do WebUI | Função de extração esperava formato árvore recursiva | Usar `extract_elements_from_raw()` com join axtree + bb.json.gz |
| `AttributeError: NoneType has no attribute 'get'` | Entradas `None` no bb.json.gz | Verificação `isinstance(box, dict)` antes de aceder campos |
| Roles não mapeados (ex: `Section`, `ListMarker`) | Roles CDP com capitalização diferente | Adicionados explicitamente a `WEBUI_TO_UNIFIED` com valor `None` |
| `On/Off Switch` perdido no Rico | Variante não listada no mapeamento | Adicionado `"On/Off Switch": "Toggle"` a `RICO_TO_UNIFIED` |
| Download do Google Drive falha | Quota de download excedida | Tentar mais tarde; ou fazer download manual e colocar em `data/webui_tmp/` |

---

## 10. Verificação da Reprodução

Após executar todos os passos, verificar:

```bash
# Contagem de imagens por split
ls data/unified/combined/train/images/ | wc -l   # ≈ 84.104
ls data/unified/combined/val/images/   | wc -l   # ≈ 10.513
ls data/unified/combined/test/images/  | wc -l   # ≈ 10.513

# Verificar data.yaml
cat data/unified/combined/data.yaml

# Verificar formato de um label
head -5 data/unified/combined/train/labels/rico_1.txt
```

Valores esperados (com tolerância de ±1% devido a variações no download/extração):
- **Total de imagens:** ~105.130
- **Total de anotações:** ~3.317.974
- **Classes cobertas:** 12/12
- **Classe mais frequente:** Text (~62%)
- **Classe menos frequente:** Toggle (~0.1%)
