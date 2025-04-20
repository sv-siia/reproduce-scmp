# Reproduce Strategic Classification in Machine Learning

## Team Members

- Samidullo Abdullaev
- Porimol Chandro
- Shokoufeh Naseri
- Anastasiia Sviridova

## Subject of Research

Our project focuses on reproducing and evaluating the methodologies presented in the paper "Strategic Classification Made Practical" by Sagi Levanon and Nir Rosenfeld. This study addresses the challenges posed by strategic behavior in classification tasks, where individuals may alter their features to receive favorable outcomes. The authors propose a learning framework that directly minimizes the "strategic" empirical risk by differentiating through the strategic responses of users.

## Approach to Solving the Problem

1. **Literature Review:** Conduct an in-depth review of existing research on strategic classification to understand the theoretical foundations and current challenges.
2. **Reproducing the Paper's Framework:** Implement the strategic empirical risk minimization framework exactly as described in the paper, ensuring consistency with the original methodology.
3. **Dataset Selection:** Identify and preprocess datasets used in the paper (if available) or select similar datasets where strategic behavior is plausible.
4. **Model Training and Evaluation:** Train classification models using both traditional and strategic approaches, following the paper’s experimental setup, and compare their performance.
5. **Validation and Analysis:** Analyze the reproduced results to assess whether they align with the findings in the paper and interpret the implications for real-world applications.

## Tools and Technologies

- **Programming Languages:** Python
- **Libraries and Frameworks:** TensorFlow or PyTorch for model
implementation, NumPy and Pandas for data manipulation, Matplotlib or Seaborn for visualization.
- **Version Control:** Git and GitHub for collaborative development and version tracking.

## How to install and run?

At first install the `uv` python package manager, see more [here](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

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

## Team Member Responsibilities

- **Porimol Chandro:** Lead the implementation of the strategic classification framework and oversee the integration of various components.
- **Samidullo Abdullaev:** Manage data collection, preprocessing, and
exploratory data analysis to ensure data quality and suitability.
**Anastasiia Sviridova:** Develop and optimize machine learning models, focusing on both traditional and strategic classification approaches.
**Shokoufeh Naseri:** Conduct literature reviews and assist in analyzing results, contributing to the interpretation and documentation of findings.

## Conclusion

By reimplementing the methodologies from `Strategic Classification Made Practical,` our project aims to bridge the gap between theoretical research and practical application in the field of strategic classification. This work will provide insights into the challenges and solutions associated with strategic behavior in machine learning models, contributing to the development of more robust and fair predictive systems.
