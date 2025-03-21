# RedGEMPy

A Python implementation of RedGEM for reduction of genome-scale metabolic models (GEMs).

## Description

RedGEMPy is a Python port of the original MATLAB-based [RedGEM](https://github.com/EPFL-LCSB/redgem) framework, which enables systematic reduction of genome-scale metabolic models for development of consistent core metabolic models. The package includes the RedGEM workflow for defining core subsystems and connecting them with minimal reaction sets, as well as LumpGEM for generating biomass building blocks through elementally balanced lumped reactions.

## Features

- Import and process genome-scale metabolic models in various formats
- Identify and connect core subsystems with minimal reaction sets
- Apply thermodynamic constraints (optional)
- Generate lumped reactions for biomass building blocks
- Post-process the reduced model to remove blocked reactions
- Export the final reduced model in JSON and MATLAB formats

## Installation

### From PyPI

```bash
pip install redgempy
```

### From Source

```bash
git clone https://github.com/EPFL-LCSB/redgempy.git
cd redgempy
pip install -e .
```

## Usage

### Basic Example

```python
from redgempy.redgem import run_redgem

# Dictionary with configuration options
config = {
    "organism": "ecoli",
    "gem_name": "iJO1366",
    "red_model_name": "core_ecoli",
    "selected_subsystems": ["Citric Acid Cycle", "Glycolysis/Gluconeogenesis"],
    "add_etc_as_subsystem": "yes",
    "l": 10,  # Max pathway length to search
    "d": 4,   # Max pathway length to include
    "start_from_min": "yes",
    "perform_lumpgem": "yes",
    "output_path": "./output",
}

# Run the RedGEM workflow
redgem_model = run_redgem(config_dict=config)

# Access the reduced model
reduced_model = redgem_model.red_model
```

### Using Configuration File

Create a YAML configuration file (`config.yaml`):

```yaml
organism: ecoli
gem_name: iJO1366
red_model_name: core_ecoli
selected_subsystems: default
add_etc_as_subsystem: yes
l: 10
d: 4
start_from_min: yes
perform_lumpgem: yes
output_path: ./output
```

Then run the workflow:

```python
from redgempy.redgem import run_redgem

redgem_model = run_redgem(config_file="config.yaml")
```

### Command Line Interface

```bash
redgempy -c config.yaml
```

## Configuration Options

| Option                        | Description                                       | Default Value   |
| ----------------------------- | ------------------------------------------------- | --------------- |
| `organism`                    | Organism name                                     | `ecoli`         |
| `gem_name`                    | GEM model name                                    | `default_gem`   |
| `red_model_name`              | Name for the reduced model                        | `reduced_model` |
| `selected_subsystems`         | Core subsystems (list or `default`)               | `default`       |
| `add_etc_as_subsystem`        | Add electron transport chain as subsystem         | `yes`           |
| `add_extracellular_subsystem` | Add extracellular subsystem                       | `automatic`     |
| `aerobic_anaerobic`           | Set conditions as aerobic or anaerobic            | `aerobic`       |
| `l`                           | Maximum pathway length to compute                 | `10`            |
| `d`                           | Maximum pathway length to include                 | `4`             |
| `start_from_min`              | Start counting from shortest path                 | `yes`           |
| `only_connect_exclusive_mets` | Only connect metabolites unique to each subsystem | `yes`           |
| `perform_lumpgem`             | Perform LumpGEM to create lumped reactions        | `yes`           |
| `impose_thermodynamics`       | Apply thermodynamic constraints                   | `no`            |
| `perform_post_processing`     | Perform post-processing                           | `no`            |
| `output_path`                 | Directory for output files                        | `./outputs`     |

## References

- RedGEM: Ataman, M., et al., "redGEM: Systematic reduction and analysis of genome-scale metabolic reconstructions for development of consistent core metabolic models". Plos Computational Biology, 2017. 13(7).
- RedGEMX: Maria Masid, Meriç Ataman and Vassily Hatzimanikatis. "redHUMAN: analyzing human metabolism and growth media through systematic reductions of thermodynamically curated genome-scale models"
- LumpGEM: Meric Ataman and Vassily Hatzimanikatis. "lumpGEM: Systematic generation of subnetworks and elementally balanced lumped reactions for the biosynthesis of target metabolites". Plos Computational Biology, 2017. 13(7).

## License

Apache License 2.0