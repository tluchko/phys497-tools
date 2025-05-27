# phys497-tools

**phys497-tools** is a set of scripts for managing and grading **PHYS 497: Senior Capstone in Physics**, a research-focused physics course at CSUN. The course emphasizes:

- Research ethics and professionalism  
- Literature review and scientific communication  
- Time management and project planning  
- Peer review and oral presentations  
- Final written reports  
- The Major Field Test (MFT) in Physics

## Scripts

- **`assign-peer-review.py`**  
  Randomizes student presentation order and/or assigns peer reviewers based on the class roster.

- **`rank-practice-problem.py`**  
  Analyzes student performance on MFT-style practice problems and compares results against national averages.

- **`grades.py`**  
  Computes final grades using Canvas exports, assignment completion data, and MFT percentile results. Produces grade output files for Canvas and CSUN’s SOLAR system.

## Usage

Each script is run from the command line. For usage details and available options, run:

```bash
python script.py -h
```
Replace `script.py` with the name of the script you want to run (e.g., `grades.py`).


## Requirements

* Python 3.7 or higher
* Python packages:
  * pandas
  * numpy
  * scipy

### Install with pip:

```
pip install -r requirements.txt
```

### Install with conda:

```
conda create -n phys497-tools --file requirements.txt
```
