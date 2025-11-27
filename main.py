from sys import argv

from simulator import RaceTrack, Simulator, plt

import matplotlib

if __name__ == "__main__":
    # matplotlib.use('tkagg')
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    raceline_path = argv[2]
    simulator = Simulator(racetrack)
    simulator.start()
    plt.show()