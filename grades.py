#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import scipy.interpolate as interp
import numpy as np
import re
import collections
import operator
import argparse

pd.options.display.width = 200
pd.options.display.max_columns = 100

numeric_map = collections.OrderedDict((    
    ('A', 4.0),  ('A-', 3.7), ('B+', 3.3), ('B', 3.0),  ('B-', 2.7),
    ('C+', 2.3), ('C', 2.0),  ('C-', 1.7), ('D+', 1.3), ('D', 1.0),
    ('D-', 0.7), ('F', 0.0)
))

def main():
    args = get_args()

    # Constants and adjustable parameters
    weights = collections.OrderedDict((
        ('Homework and Participation Grade Point', 0.15),
        ('Final Presentation', 0.425),
        ('Final Report', 0.425)
    ))

    mft_thresholds = collections.OrderedDict((
        ('boost', 30),
        ('penalty', 0)
    ))

    final_thresholds = collections.OrderedDict((
        (3.7, 4.0),
        (3.3, 3.7),
        (3.0, 3.3),
        (2.7, 3.0),
        (2.3, 2.7),
        (2.0, 2.3),
        (1.7, 2.0),
        (1.3, 1.7),
        (1.0, 1.3),
        (0.7, 1.0),
        (0.0, 0)
    ))

    professionalism_thresholds = collections.OrderedDict((
        ('Preparation', collections.OrderedDict(((4., 1), (3., 2), (2., 3), (1., 4), (0., 5)))),
        ('Participation', collections.OrderedDict(((4., 2), (3., 4), (2., 6), (1., 8), (0., 10)))),
        ('Practice', collections.OrderedDict(((4., 1), (3., 1), (2., 2), (1., 3), (0., 4)))),
        ('Peer-Review', collections.OrderedDict(((4., 1), (3., 2), (2., 3), (1., 4), (0., 5))))
    ))

    canvas = pd.read_csv(args.canvas_input)
    canvas = canvas.drop(canvas.index[-1])

    # Professionalism
    ## Preparation
    prep_cols = find_columns(canvas, [
        "Anatomy of a research paper", "Next Action", "Schedule your recurring meeting with your advisor",
        "Next Action Followup", "Read the Whitesides paper", "Read the Schön Report",
        "Essay for next year's students", "Select five CSUNposium talks to attend"
    ])
    canvas[find_columns(canvas, ['Preparation'])[0]] = count_complete_assignments(canvas, prep_cols)

    ## Peer Review
    peer_cols = find_columns(canvas, [
        "Project Outline #1 - peer review", "Practice Presentation #1 - peer review", "Project Outline #2 - peer review",
        "Project Outline #3 Peer Review", "CSUNposium Reviews", "Final Report Draft #1 - peer reviews",
        "Practice Presentation #2 - peer reviews", "Final Report Draft #2 - peer reviews"
    ])
    canvas[find_columns(canvas, ['Peer-Review'])[0]] = count_complete_assignments(canvas, peer_cols)

    ## Participation
    part_cols = find_columns(canvas, [
        "Week 1 Participation", "Week 2 Participation", "Week 3 Participation", "Week 4 Participation",
        "Week 5 Participation", "Week 6 Participation", "Week 7 Participation", "Week 8 Participation",
        "Week 10 Participation", "Week 11 Participation", "Week 12 Participation", "Week 13 Participation",
        "Week 14 Participation", "Week 15 Participation", "Week 16 Participation"
    ])
    canvas[find_columns(canvas, ['Participation'])[0]] = count_complete_assignments(canvas, part_cols)

    ## Practice
    prac_cols = find_columns(canvas, [
        "Create your elevator pitch", "Project Outline #1", "Practice Presentation #1", "Project Outline #2",
        "Project Outline #3", "Final Report Draft #1", "Practice Presentation #2", "Final Report Draft #2"
    ])
    canvas[find_columns(canvas, ['Practice'])[0]] = count_complete_assignments(canvas, prac_cols)

    ## Finalize professionalism grade
    homework_col = find_columns(canvas, ['Homework and Participation Grade Point'])[0]
    canvas[homework_col] = (
        calculate_professionalism_grade(canvas, find_columns(canvas, ['Preparation', 'Participation', 'Practice', 'Peer-Review']), professionalism_thresholds))

    # Calculate final grades without MFT
    canvas = canvas.set_index('Student')
    final_col = find_columns(canvas, ['Final'])[0]
    canvas[final_col] = (
        canvas[find_columns(canvas, weights.keys())].apply(pd.to_numeric, errors='coerce').astype(float) @ list(weights.values()))
    canvas[final_col] = canvas[final_col].apply(lambda x: apply_grade_threshold(x, final_thresholds))

    # Apply MFT adjustments
    mft_col = find_columns(canvas, ['ETS Major Field Test'])[0]
    boost_mask = canvas[mft_col] > mft_thresholds['boost']
    canvas.loc[boost_mask, final_col] = canvas.loc[boost_mask, final_col].apply(lambda x: mod_grade_by_one(x, operator.add))
    drop_mask = canvas[mft_col] < mft_thresholds['penalty']
    canvas.loc[drop_mask, final_col] = canvas.loc[drop_mask, final_col].apply(lambda x: mod_grade_by_one(x, operator.sub))

    # output results
    canvas.reset_index()[['Student', 'ID', 'SIS User ID', 'SIS Login ID', 'Section']
        + find_columns(canvas, ['Preparation', 'Practice', 'Participation', 'Peer-Review', 'Homework and Participation Grade Point', 'ETS Major Field Test', 'Final'])]        .to_csv(args.canvas_output, index=False)

    canvas.loc[:, 'Final Letter Grade'] = canvas[final_col].apply(lambda x: next((k for k, v in numeric_map.items() if v == x), None))
    canvas['SIS User ID'] = canvas['SIS User ID'].astype(str).str.strip('.0')
    canvas.reset_index()[['SIS User ID', 'Final Letter Grade']].to_csv(args.solar_output, index=False)

def get_args():
    """
    Parses and returns command-line arguments for calculating final grades for PHYS 497.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing the input and output file paths.
    """
    parser = argparse.ArgumentParser(description="""Calculate final grades for PHYS 497.
Instructions
1. Download the complete grades from Canvas
2. Download the `Individual Student Reports` -> `Score Reports` from the MFT admin portal.  
   This is a PDF and is the only way to get percentile scores for the current cohort.
3. Manually enter the MFT percentile scores in the `ETS Major Field Test` column.
4. Modify the thresholds in this files as needed
5. Run this script.""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--canvas-input', required=True, help='Input CSV file from Canvas')
    parser.add_argument('--canvas-output', required=True, help='Output CSV file for Canvas')
    parser.add_argument('--solar-output', required=True, help='Output CSV file for SOLAR')
    return parser.parse_args()

def find_columns(df, names):
    r"""
    Finds and returns the column names in a DataFrame that match the given names followed by (\d+).

    Parameters:
    df (pandas.DataFrame): The DataFrame to search for columns.
    names (list of str): A list of names to match against the column names in the DataFrame. 
                         Each name is expected to match a column name followed by ' (\d+)'.

    Returns:
    list of str: A list of column names that match the given names.

    Raises:
    ValueError: If no columns are found for a given name or if multiple columns match a given name.

    Example:
    >>> find_columns(canvas, ['Select five CSUNposium talks to attend'])
    ['Select five CSUNposium talks to attend (2211230)']
    """
    return [col for name in names for col in df.columns if re.match(name + r' \(\d+\)', col)]

def count_complete_assignments(df, columns):
    """
    Counts the number of complete assignments for each row in the specified columns of a DataFrame.

    The function assumes that a value > 0 in a column indicates a complete assignment.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the assignment data.
    columns (list of str): A list of column names to evaluate for completeness.

    Returns:
    pandas.Series: A Series containing the count of complete assignments for each row.

    Example:
    >>> count_complete_assignments(canvas, ['Week 1 Participation (2211279)', 'Week 2 Participation (2211287)'])
    0    0
    1    2
    2    1
    dtype: int64
    """
    df = df[columns].copy().apply(pd.to_numeric, errors='coerce').map(lambda x: 1 if x > 1 else x)
    return df.sum(axis=1)

def calculate_professionalism_grade(df, columns, thresholds):
    """
    Calculates the professionalism grade for each student based on the given columns and thresholds.

    This function computes a numeric grade for each student by:
    1. Calculating the number of missing assignments for each category.
    2. Interpolating the numeric grade based on the thresholds for each category.
    3. Computing the final numeric grade using an anchored grading system.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the assignment data.
    columns (list of str): A list of column names to evaluate for professionalism grading.
    thresholds (OrderedDict): A dictionary mapping categories to their respective thresholds for grading.

    Returns:
    pandas.Series: A Series containing the professionalism grade for each student.

    Example:
    >>> calculate_professionalism_grade(canvas, ['Preparation', 'Participation', 'Practice', 'Peer-Review'])
    2    3.7
    3    4.0
    4    3.3
    dtype: float64
    """
    def compute_final(grades: pd.Series) -> float:
        """
        Compute the anchored final numeric grade from a Series of four category scores:
        - Anchor = minimum of the four scores.
        - count_next = number of categories >= (anchor + 1.0).
        - Raw score:
            >=2 at next level → anchor + 0.7
            =1 at next level  → anchor + 0.3
            else             → anchor
        - Snap down to nearest allowed step in the 0.0, 0.3, 0.7, 1.0, …, 4.0 scale.
        """
        anchor = grades.min()
        count_next = (grades >= anchor + 1.0).sum()
        if count_next >= 2:
            raw = anchor + 0.7
        elif count_next == 1:
            raw = anchor + 0.3
        else:
            raw = anchor

        allowed = sorted(numeric_map.values())
        return max(v for v in allowed if v <= raw + 1e-8)

    df = df[columns].copy()
    df.loc[1:, :] = df.loc[1, :] - df.loc[1:, :]
    for col in df.columns:
        category = re.sub(r' \(\d+\)', '', col)
        threshold = thresholds[category]
        lin_interp = interp.interp1d(list(threshold.values()), list(threshold.keys()), fill_value=(4., 0.), bounds_error=False)
        df[col] = lin_interp(df[col])
    
    return df.apply(compute_final, axis=1)

def apply_grade_threshold(x, thresholds):
    """
    Apply grade thresholds to a numeric value and return the corresponding grade.

    Parameters:
    x (float or int): The numeric value to be graded. If NaN, it will be returned as is.
    thresholds (dict): A dictionary where keys are numeric threshold values and 
                       values are the corresponding grades. The thresholds should 
                       be sorted in descending order for proper evaluation.

    Returns:
    The grade corresponding to the first threshold that `x` meets or exceeds. 
    If `x` is NaN, it is returned unchanged.
    """
    if pd.isna(x): return x
    for v in thresholds: 
        if x >= v: return thresholds[v]

def mod_grade_by_one(x, op):
    """
    Modify a grade by one step in a predefined set of grade values.

    Grades are expected to be numeric value of the ± system: [0.0, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0].

    Parameters:
    x (float): The current grade value. If `x` is NaN, it will be returned as is.
    op (function): A function that takes two arguments: the current index of `x` in the grade values 
                   list and the step (1). This function determines how the index is modified 
                   (e.g., increment or decrement).

    Returns:
    float: The modified grade value, constrained within the range of the predefined grade values. 
           If `x` is NaN, it is returned unchanged.
    """
    values = sorted(set([0.0, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0]))
    return values[max(min(op(values.index(x), 1), len(values)-1), 0)] if not pd.isna(x) else x

if __name__ == '__main__':
    main()
