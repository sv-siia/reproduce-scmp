# Reproduce Strategic Classification in Machine Learning

## Team Members

- Samidullo Abdullaev
- Porimol Chandro
- Anastasiia Sviridova
- Shokoufeh Naseri

## Team Member Responsibilities

- **Porimol Chandro:** Lead the (reproducible) implementation design plans.
- **Samidullo Abdullaev:** Data collection, preprocessing, and exploratory data analysis.
- **Anastasiia Sviridova:** Reproduce and optimize machine learning models.
- **Shokoufeh Naseri:** Contributed to code refactoring, Conducted literature reviews, Git workflow management, and implementation debugging to ensure successful reproduction of the original results.

## Subject of Research

Our project focuses on reproducing and evaluating the methodologies presented in the paper "Strategic Classification Made Practical" by Sagi Levanon and Nir Rosenfeld. This study addresses the challenges posed by strategic behavior in classification tasks, where individuals may alter their features to receive favorable outcomes. The authors propose a learning framework that directly minimizes the "strategic" empirical risk by differentiating through the strategic responses of users.

## Approach to Solving the Problem

1. **Literature Review:** Conduct an in-depth review of existing research on strategic classification to understand the theoretical foundations and current challenges.
2. **Reproducing the Paper's Framework:** Implement the strategic empirical risk minimization framework exactly as described in the paper, ensuring consistency with the original methodology.
3. **Dataset Selection:** Identify and preprocess datasets used in the paper (if available) or select similar datasets where strategic behavior is plausible.
4. **Model Training and Evaluation:** Train classification models using both traditional and strategic approaches, following the paper’s experimental setup, and compare their performance.
5. **Validation and Analysis:** Analyze the reproduced results to assess whether they align with the findings in the paper and interpret the implications for real-world applications.

## Tools and Technologies

- **Programming Languages:** Python 3.12.*
- **Package Dependency Manager:** [UV](https://docs.astral.sh/uv/)
- **Libraries and Frameworks:** PyTorch, Numpy and Pandas for data manipulation, Matplotlib.
- **Version Control:** Git and [GitHub](https://github.com/sv-siia/reproduce-scmp) for collaborative development and for source code version control.

## How to install and run?

At first install the `uv` python package manager, see more [here](https://docs.astral.sh/uv/getting-started/installation/).

Clone the code repository as follows

```bash
git clone git@github.com:sv-siia/reproduce-scmp.git

# now change directory to the repository
cd reproduce-scmp
```

Install Python dependencies using `uv` package manager

```bash
uv sync
```

## Developer Installation (Editable mode)

To install the project as a developer package (editable mode), please follow these steps:

1. Ensure you have installed the `uv` Python package manager. For installation instructions, refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. Install the project in editable mode by running the following command in the project directory:

   ```bash
   uv pip install --editable .
   ```

   This command will:
   - Install the project as a package in editable mode.
   - Allow you to make changes to the code and immediately use them without reinstalling.
   - Install all required dependencies specified in the `pyproject.toml` file.

3. Verify the installation by running the CLI command:

   ```bash
   scmp --help
   ```

   This should displaying the help message for the `scmp` CLI tool, confirming that the package is installed and ready for development.

   ```bash
   Usage: scmp [OPTIONS] COMMAND [ARGS]... 
                                   
   CLI for the reproduce-scmp project.                                                                                                                                                                                                                                                             
   ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
   │ --install-completion          Install completion for the current shell.                                                   │
   │ --show-completion             Show completion for the current shell, to copy it or customize the installation.            │
   │ --help                        Show this message and exit.                                                                 │
   ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
   ╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
   │ evaluate   Evaluate the trained models.                                                                                   │
   │ train      Train specific models.                                                                                         │
   ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
   ```

   ```bash
   scmp train --help
   ```

   The command `scmp train --help` shows the help message for the train subcommand of the scmp CLI tool. It provides an overview of the available training commands for different models (e.g., RNN, Recourse, etc.) and brief descriptions.

   ```bash
   Usage: scmp train [OPTIONS] COMMAND [ARGS]...
   
   Train specific models.
   
   ╭─ Options ──────────────────────────────────────╮
   │ --help          Show this message and exit.    │
   ╰────────────────────────────────────────────────╯
   ╭─ Commands ─────────────────────────────────────╮
   │ rnn        Train the RNN model.                │
   │ recourse   Train the Recourse model.           │
   │ utility    Train the Utility model.            │
   │ batched    Train the Batched model.            │
   │ burden     Train the Burden model.             │
   │ vanila     Train the Vanila model.             │
   │ manifold   Train the Manifold model.           │
   ╰────────────────────────────────────────────────╯
   ```

## Example

### RNN model training

if everything went well as described above, run the following command to train the RNN model

```bash
scmp train rnn dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/rnn
```

**In console you will see something similar as follows**

```bash
 solves will not be faster than the first one. For more information, see the documentation on Disciplined Parametrized Programming, at https://www.cvxpy.org/tutorial/dpp/index.html
  warnings.warn(DPP_ERROR_MSG)
model saved!
training time: 11.583229064941406 seconds
---------- training non-strategically----------
.venv/lib/python3.12/site-packages/cvxpy/expressions/expression.py:674: UserWarning: 
This use of ``*`` has resulted in matrix multiplication.
Using ``*`` for matrix multiplication has been deprecated since CVXPY 1.1.
    Use ``*`` for matrix-scalar and vector-scalar multiplication.
    Use ``@`` for matrix-matrix and matrix-vector multiplication.
    Use ``multiply`` for elementwise multiplication.
This code path has been hit 15 times so far.

  warnings.warn(msg, UserWarning)
model saved!
```

## Conclusion

By reimplementing the methodologies from `Strategic Classification Made Practical,` our project aims to bridge the gap between theoretical research and practical application in the field of strategic classification. This work will provide insights into the challenges and solutions associated with strategic behavior in machine learning models, contributing to the development of more robust and fair predictive systems.
