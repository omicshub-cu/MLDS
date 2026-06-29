import sys
sys.path.insert(0, '.')  # Add current folder to path

from synthselector import SyntheticEvaluator
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Test it works
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={"target": "label"})
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

# Create fake synthetic
synth = train.copy()
synth.iloc[:, :-1] += np.random.normal(0, 0.1, synth.iloc[:, :-1].shape)

# Run
evaluator = SyntheticEvaluator(train, test)
evaluator.add("test", synth)
result = evaluator.run()
result.print_report()