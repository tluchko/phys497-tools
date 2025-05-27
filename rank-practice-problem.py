import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    args = get_args()

    report = pd.read_csv(args.item_report)
    if args.question_list:
        questions = pd.read_csv(args.question_list, header=None, names=['names'])
        report['Question Title'] = questions
        if args.p_values:
            p_values = pd.read_csv(args.p_values)
            print(p_values)
            report = report.merge(p_values, how="left",
                                  left_on="Question Title",
                                  right_on= "Question")
    report = report.sort_values('Correct Student Count')
    if "P" in report.columns:
        print(report[['Question Id', 'Question Title', 'Correct Student Count', "P"]])
    else:
        print(report[['Question Id', 'Question Title', 'Correct Student Count']])

    if args.plot:
        if not (args.question_list and args.p_values):
            raise(RuntimeError,
                  "A question list and p-values are needed for plotting.")
        plot(args.plot, report)

def get_args():
    '''reads in and returns command line arguments'''
    parser = argparse.ArgumentParser(description='''
Reads in Canvas item report and ranks questions by their difficulty''')
    parser.add_argument('item_report',
                        help = 'Canvas item report')
    parser.add_argument('question_list',
                        nargs='?',
                        help = 'Ordered list of question names to be applied. Sets the values of "Question Title"')
    parser.add_argument('p_values',
                        nargs='?',
                        help = 'CSV file with "Question" and "P" columnns.  This is merged with the item report by matching "Question" with "Question Title".')
    parser.add_argument('--plot',
                        nargs="?",
                        const='show',
                        help = 'Plot "Correct Student Count" vs P value. Optionally provide a file name to plot to')
    return parser.parse_args()
    
def plot(filename, df):
    plt.scatter(df.P, df["Correct Student Count"], marker="o")
    for index, row in df.iterrows():
        plt.annotate(row["Question Title"],
                     (row.P, row['Correct Student Count']),
                     rotation=90,
                     xytext=(0,5),
                     textcoords='offset points')
    plt.xlabel("P+ value")
    plt.ylabel("Number of students correct")
    plt.tight_layout()
    if filename == "show":
        plt.show()
    else:
        plt.savefig(filename)

if __name__ == "__main__":
    main()
