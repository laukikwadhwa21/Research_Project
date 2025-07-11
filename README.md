# Research_Project
Code Generation Approach for Question-Answering Over Tabular Data 

# AlphaPro at SemEval-2025 Task 8: A Code Generation Approach for Question-Answering over Tabular Data

## Description
This repository presents the code implementation for the approach by team AlphaPro at SemEval 2025 - Task 8. The approach is to eventually generate Python code for question-answering, given a dataset.

## Table of Contents
* [Requirements](#requirements)
* [Directory Structure](#directory-structure)
* [Architecture](#architecture)
* [Results](#results)
* [Citation](#citation)
* [Contact](#contact)

## Requirements
Using the Command R Model from Cohere.

```
pip install cohere
```

Using `load_dataset` from `datasets` library for loading the QA dataset.

```
pip install datasets
```

## Parameters
* Input:
   * Question: `str`
   * Dataset: `pandas.DataFrame`
* Output:
   * Original Question: `str`
   * Paraphrased Question: `str`
   * Code: `str`
   * Expected Answer Type: `str`
   * Output (Actual Answer): `str`

## Directory Structure

```
├── notebooks/
│   ├── AlphaProQA.ipynb       # Main notebook with code explaining approach
│   └── EvalPlot.ipynb         # Colab notebook used for generating and saving graphs
├── src/
│   ├── AlphaProQA.py          # .py equivalent of notebook with class created for importing
│   ├── runner.py              # Running the model for saving the outputs to CSV files
│   ├── plotter.py             # Plotting the result graphs
│   └── evalSetGen.py          # Manually formed questions for further performance insight
├── results/
│   ├── Results_1.csv          # Main output files along with related information
│   └── graded_qa.csv          # Complexity graded questions
└── README.md
```

## Architecture
Question Answering Logic:
* Step 1:
   * Get the dataset schema from the pandas.DataFrame object of the dataset.
* Step 2:
   * Rewrite the given question using an LLM so that the paraphrased question now uses the table schema in its wording.
   * Predict the expected answer type.
* Step 3:
   * Generate Python code (fill the function given in the prompt) for answering the paraphrased question, given the dataset, schema and expected answer type.
* Step 4:
   * Extract the Python function into the current namespace for execution. This function is deleted after execution for clean environment.
* Step 5:
   * Run the function and report answer or error accordingly.

## Results
The system is able to answer questions upto 75% accuracy%.

## Citation
To be added

## Contact
For any questions or feedback, please contact:
- Anshuman Aryan: anshuman.aryan24@gmail.com
- Laukik Wadhwa: laukikwadhwa@gmail.com
- Kalki Eshwar D: kalkieshward@gmail.com
- Aakarsh Sinha: aakarshsinha.in@gmail.com
- Durgesh Kumar: durgesh.nlpai@gmail.com
