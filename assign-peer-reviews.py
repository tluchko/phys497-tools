import pandas as pd
import argparse
import itertools
import numpy as np


def main():
    args = get_args()
    df = pd.read_csv(args.roster)
    df = df[['Student']].set_index('Student')
    try:
        df = df.drop(index=['Student, Test', '    Points Possible'] )
    except:
        pass
    df = df[df.index.notnull()]
    df = df.reset_index()
    if not args.ordered:
        df = df.sample(frac=1)
    reviews = assign_reviews(df, args.N, args.presentation_mode)

    reviewers = reviews.iloc[:,[0]].reset_index().set_index(reviews.columns[0])


    for column in reviews.columns[1:]:
        df = reviews[[column]].reset_index().set_index(column)
        reviewers = pd.merge(reviewers, df, left_index=True, right_index=True)
    reviewers = reviewers.sort_index()

    if args.sorted:
        reviews = reviews.sort_index()
    if args.csv:
        reviews.to_csv(args.csv)
        reviewers.to_csv(args.csv, mode = 'a')
    else:
        print('---------------')
        print('Presenter Order')
        print('---------------')
        print(reviews)
        print('---------------')
        print('Reviewer Order')
        print('---------------')
        print(reviewers)
        
    
def get_args():
    '''Get the command-line arguments'''
    parser = argparse.ArgumentParser(description='''
Assign students to reviewer N of their peers.''')
    parser.add_argument('-N', '--N', type=int, default=3,
                        help = 'Number of reviewers per student')
    parser.add_argument('--presentation_mode', action='store_true',
                        help="Removes permutations where the reviewer is reviewing the first two people before them in the order before them.")
    parser.add_argument('--ordered', action='store_true',
                        help="Legacy feature. Maintains the original order of the roster throughout the process and is less randomized.")
    parser.add_argument('--sorted', action='store_true',
                        help="Final list of reviewers is sorted alphabetically..")
    parser.add_argument('roster',
                        help = 'Canvas grade roster')
    parser.add_argument('csv', nargs='?',
                        help = 'Output results to a CSV file instead of printing to screen.')
    return parser.parse_args()

def assign_reviews(df, n, presentation_mode):
    '''Assign N reviews per student such that everybody gets evenly
    reviewed and no one reviews themselves.
    
    Args:
        df : dataframe with just a Student column.
        n: number of reviews
        presentation_mode: remove instances where students are
            assigned either of the two students before them in the order.
    Returns:
        dataframe with Student and n columns of reviews.'''

    end=1
    start=0
    if presentation_mode:
        start=1
        end=2

    for i in range(start, len(df)-end):
        df[i] =  np.roll(df.Student.values, shift=i+1)
    df = df.set_index('Student')
    # print(df)
    reviews = df[np.random.choice(df.columns, n, replace=False)]
    reviews.columns = range(0,n)
    return reviews


    
def contains_self(sample, df):
    for col in df.columns:
        if (sample.values == df[[col]].values).any():#(axis=None):
            # print(sample.merge(df, left_index=True, right_index=True))
            # print(sample == df[[col]])
            # print((sample == df[[col]]).any(axis=None))
            return True
    return False

if __name__ == "__main__":
    main()
