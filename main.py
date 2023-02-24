# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import ex2 as ex

def import_dataset():
    filename = 'data/simulation-pose-landmark.g2o'
    graph = ex.read_graph_g2o(filename)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import_dataset()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
