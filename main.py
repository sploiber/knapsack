import hybrid
import pandas as pd
from knapsack import Knapsack
import click
@click.command()
@click.argument('data_file_name')
@click.argument('max_weight', default=1.)
def main(data_file_name, max_weight):

    # check that the user has provided data file name, and maximum weight
    # which the knapsack can hold
    try:
        with open(data_file_name, "r") as myfile:
            input_data = myfile.readlines()
    except IndexError:
        print("Usage: knapsack.py: <data file> <maximum weight>")
        exit(1)
    except IOError:
        print("knapsack.py: data file <" + data_file_name + "> missing")
        exit(1)

    try:
        W = float(max_weight)
    except IndexError:
        print("Usage: knapsack.py: <data file> <maximum weight>")
        exit(1)

    if W <= 0.:
        print("Usage: knapsack.py: <maximum weight> must be positive")
        exit(1)

    # parse input data
    df = pd.read_csv(data_file_name, header=None)
    df.columns = ['name', 'cost', 'wt']

    # create the Knapsack object
    K = Knapsack(df['name'], df['cost'], df['wt'], W)

    # Obtain the knapsack BQM
    bqm = K.get_bqm()

    # Set up workflow for dwave_hybrid
    iteration = hybrid.RacingBranches(hybrid.SimulatedAnnealingProblemSampler(),
                                      hybrid.InterruptableTabuSampler()) | hybrid.ArgMin()
    # arbitrarily using 2.0 seconds
    workflow = hybrid.Loop(iteration, max_time=2.0)

    # Set up dwave_hybrid state
    state = hybrid.State.from_problem(bqm)

    # Run dwave_hybrid and obtain the result
    result = workflow.run(state).result()

    # Obtain the lowest-energy solution found by dwave_hybrid, and obtain its
    # representation in terms of the object names
    print(K.get_names(result.samples.record['sample'][0]))


if __name__ == '__main__':
    main()
